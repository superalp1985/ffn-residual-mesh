# MiniMax H3 ComfyUI 手机/平板 FFN 集群设计

版本：0.1（2026-09-05）
状态：架构设计与分析模拟完成；尚未连接真实 Android 设备。

## 1. 目标

在不改动 ComfyUI 工作流语义的前提下：

- 主机 RTX 4070 保留 attention、SageAttention、residual、SwiGLU 和最终合并；
- 手机/平板保存冷启动编译后的 FFN 基项表，承担 gate/up 的主项聚合；
- TeaCache 继续负责跨 timestep 的整次 DiT forward 复用；
- 网络只使用持久连接、二进制 tile 和异步回传；
- 任一 worker 超时都回退主机精确路径。

这不是把 ComfyUI 拆成几十个 RPC 节点，而是在模型 forward 内增加一个可选的 FFN worker backend。

## 2. 本地 MiniMax H3 事实

本机环境已经具备：

- ComfyUI 0.30.0；
- Python 3.12.10；
- torch 2.9.1+cu130，torch.version.cuda=13.0；
- NVIDIA GeForce RTX 4070 Laptop GPU；
- MiniMax H3 主模型 minimax_h3_fl2va_pruned_int8_convrot.safetensors；
- MiniMax H3 TeaCache 和 KJNodes。

从本地 safetensors 元数据读取到的 DiT block 形状：

~~~text
50 blocks
hidden = 5376
mlp.fc1 = [28672, 5376] int8
mlp.fc2 = [5376, 14336] int8
~~~

fc1 的 28672 是 2 x 14336，符合 gated FFN。现有 8G 工作流已经把 SageAttention -> TeaCache -> KSampler 串起来，FFN 集群应作为这条链内部的可选实现，而不是替代 TeaCache 或 SageAttention。

## 3. TeaCache 与 FFN 集群的关系

TeaCache 在每个采样 timestep 判断是否复用上一次完整模型输出：

~~~text
rel_l1(x_t, x_{t-1})
accumulated_delta < threshold
=> reuse previous model output
~~~

因此：

- 复用 step 不需要手机网络，也不需要 FFN residual；
- 只有 real forward step 才触发手机广播、GPU residual 和合并；
- 音频开启时仍应使用 audio guard；长视频有声音时不要为了集群强行提高复用率；
- SageAttention 保持在中心 GPU，手机不参与 attention。

## 4. 数据流

### 4.1 冷启动

~~~text
H3 int8 fc1/fc2
    -> 完整扫描一次
    -> 每 block/output tile 编译 base table、residual tile、scale、checksum
    -> 主机 RAM + worker 本地存储
~~~

手机不需要保存原始 safetensors 的全部格式，只需保存版本化 artifact。artifact 必须绑定模型哈希、block、量化格式、tile 大小和公式版本。

### 4.2 real forward step

~~~text
ComfyUI model wrapper
    -> 生成 token tile descriptor
    -> 广播给手机/平板 worker
    -> worker 计算 gate/up base rows
    -> worker 返回有序 base tile
    -> GPU residual kernel 计算 residual tile
    -> 中心 GPU 合并 gate/up
    -> 原始 SwiGLU
    -> fc2/down、attention、SageAttention 继续在 GPU
~~~

每个包至少包含：

~~~text
magic, protocol_version, run_id, step_id, block_id,
tile_id, row_start, row_count, dtype, payload_bytes, checksum
~~~

必须使用持久 TCP/QUIC/USB 连接、流水线预取和 deadline；不能每个 tile 新建 RPC。

## 5. 为什么 H3 比 2B decode 难

124 帧、832x480 的默认工作流约有：

~~~text
video rows = 14,430
audio rows = 414
text rows  = 512
packed rows = 15,356
~~~

单个 real step 的 gate/up base 如果按 fp16 返回：

~~~text
15,356 x 28,672 x 2 bytes
≈ 840 MiB
~~~

这就是关键瓶颈。手机计算本身只需几十毫秒的理想内存带宽预算，但 1 Gb/s 网络传完 base 约需 7.0 秒。普通 Wi-Fi 手机集群因此不能直接承担“精确 gate/up 回传”的在线主路径。

## 6. 模拟结果

模拟脚本：

~~~text
python scripts/simulate_comfyui_phone_ffn.py \
  --width 832 --height 480 --frames 124 \
  --steps 20 --tea-real-steps 8 \
  --network-gbps 1 --phone-return gate_up_exact \
  --out results/comfyui_h3_phone_exact_1gbps.json
~~~

默认参考是本机日志中的 20-step H3 采样约 12.33 s/real step，TeaCache 假设 8 次 real forward、12 次 reuse。结果：

| 模式 | 网络 | 每 real step base 回传 | phone 分支 | 总采样+初始化估计 |
|---|---:|---:|---:|---:|
| 精确 gate/up | 1 Gb/s | 839.8 MiB | 7.11 s | 116.7 s |
| 精确 gate/up | 10 Gb/s | 839.8 MiB | 0.74 s | 114.2 s |
| hidden 近似 | 1 Gb/s | 157.5 MiB | 1.39 s | 114.2 s |

这说明：

1. 手机数量从 1 增加到 64 时，1 Gb/s 场景仍被单条网络回传主导；
2. 10 Gb/s 以上，手机分支可以隐藏在 GPU real-step 分支之后；
3. hidden 近似虽然网络小很多，但它不再是 gate/up 精确合并，必须单独报告误差并保留回退；
4. TeaCache 是关键乘数：只做 8 个 real step 时，手机和 residual 交通量约为完整 20-step 的 40%。

## 7. 能否用于跑 ComfyUI

**可以作为 ComfyUI 的可选后端运行，但不能先假设普通 Wi-Fi 手机集群会加速。**

推荐的实际部署层次：

1. **第一阶段：主机 loopback worker**
   - 多进程模拟手机；
   - 持久 socket；
   - row-sharded fc1 artifact；
   - 测 packet、deadline、顺序和 fallback。
2. **第二阶段：同机/USB 平板**
   - 先用 10 GbE、USB 3.x 或局域网有线链路；
   - 只接 124 帧、无音频的 TeaCache 工作流；
   - 记录真实 base return 与 GPU 等待。
3. **第三阶段：真实手机池**
   - 手机只做稳定的 base tile；
   - worker 断线立即回退 GPU；
   - 音频开启时降低 TeaCache 复用或关闭集群近似。
4. **第四阶段：ComfyUI 节点封装**
   - MiniMaxH3PhoneFFNBackend 模型包装节点；
   - 不修改 ComfyUI 核心；
   - 与 PathchSageAttentionKJ、MiniMaxH3TeaCache 保持可组合。

## 8. 第一版验收指标

必须同时记录：

~~~text
network payload bytes / real step
base return MiB / real step
GPU residual H2D bytes
phone ready deadline miss rate
GPU copy/compute overlap
TeaCache real/reuse count
audio guard forced-real count
exact fallback rate
最终视频和音频质量
~~~

如果 exact gate/up 模式在目标链路上不能把 phone ready 隐藏到 GPU residual 分支之后，就不应把手机放进默认关键路径；可先用于冷启动编译、预热 artifact、批量 prefill 或非实时队列。

## 9. 结论

MiniMax H3 证明了这条路线的边界：

- ComfyUI、CU130、TeaCache、SageAttention 可以共存；
- 手机集群可以扩展 FFN 基项的存储和 CPU 带宽；
- 对 H3 这种 1.5 万 packed token 的视频 forward，精确 gate/up 回传会把网络变成主瓶颈；
- 真正可行的方向是更强的基项编码、分块回传、GPU 侧 residual 复用和高带宽链路；
- 在这些条件满足前，手机集群应作为可选 backend，保留 GPU 精确 fallback，而不是强制替换 ComfyUI 的模型执行路径。

