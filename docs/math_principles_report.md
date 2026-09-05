# FFN Base + Residual：数学原理与 Qwen3.8-27B 适配报告

版本：0.1（2026-09-05）
状态：数学与单层桥接原型已验证；Qwen3.8-27B 尚未完成真实权重编译和端到端实测。

## 摘要

本项目研究一种资源置换型 FFN 推理路径：冷启动时完整扫描量化权重，把它们编译成主机内存中的分层公式、部分和、残差包和回退索引；运行时由 CPU 负责轻量路由/聚合，GPU 只接收残差 tile、完成高密度算子，并在本层内合并得到原始 FFN 输出。

目标不是减少 FLOPs，而是让每个 H2D 字节触发更多 GPU 运算，减少 GPU 等待 PCIe/显存数据的时间。主机内存可以长期保存较大的预展开产物；显存峰值不要求低于原始实现，只要不 OOM 且能用双缓冲/预取维持连续计算。

当前结论：

1. 在量化整数域内，分块基项 + 完整残差可以严格重建原始点积。
2. SwiGLU 必须先分别重建 gate 和 up，再执行原始非线性；不能跨非线性直接相加。
3. 数学恒等式本身不保证带宽收益。只有当基项被编译为有限聚合/公式，而不是运行时顺序扫描另一份稠密权重流时，才满足本项目的运行时目标。
4. Qwen3.8-27B 的 FFN 结构与该方法兼容，但必须按其 5120 x 17408、64 层和实际量化格式重新编译；现有 Qwen3.5-2B artifact 不能直接复用。

## 1. 研究边界

本报告只讨论 Transformer 前馈层（FFN/SwiGLU）。Attention、KV cache、MoE 专家路由和完整推理框架重写不在当前验证范围内。每层独立选择公式和回退策略，不要求全模型使用同一种拆分。

运行时边界：

~~~text
冷启动：原始 Q4/Q8 权重 -> 完整扫描 -> 编译 artifact 写入主机 RAM
运行时：artifact -> CPU 基项聚合 + GPU 残差 -> 本层合并 -> SwiGLU/down
回退：必要时重新读取原始权重；回退次数必须单独计数
~~~

运行阶段不得顺序扫描原始权重，也不得把同样稠密的“展开高位权重流”换个名字再扫描。基项必须是有限公式、有限状态表、块部分和或其他不枚举原始每权重码的结构。

## 2. FFN 与 SwiGLU 的正确合并位置

以 Qwen 系列常见的 gated FFN 为例：

~~~text
g = W_gate x
u = W_up   x
h = SiLU(g) ⊙ u
y = W_down h
~~~

拆分后应保持：

~~~text
g = g_base + g_residual
u = u_base + u_residual
h = SiLU(g) ⊙ u
y = W_down h
~~~

不能使用下面的非法替换：

~~~text
SiLU(g_base) ⊙ u_base
  + SiLU(g_residual) ⊙ u_residual
~~~

因为一般情况下：

~~~text
SiLU(g_base + g_residual) ⊙ (u_base + u_residual)
!= SiLU(g_base) ⊙ u_base + SiLU(g_residual) ⊙ u_residual
~~~

所以 gate 与 up 的残差必须先在 pre-activation 空间合并，再执行原始 SiLU 和逐元素乘法。down 投影可以选择整层常驻、分块流式或再次做基项/残差拆分，但必须在 h 已经得到后处理。

## 3. Q4_K 整数域的精确投影

对一个输出行 j、量化组 g 和组内位置 i，Q4_K 解码权重写成：

~~~text
w[j,g,i] = a[j,g] * q[j,g,i] - c[j,g]
~~~

其中 q 是 [0, 15] 的 4-bit code，a 与 c 是冷启动时读取的 scale/offset。将输入写成 x[g,i] = sx[g] * z[g,i]，定义：

~~~text
S[g]   = sum_i z[g,i]
P[j,g] = sum_i z[g,i] * q[j,g,i]
~~~

则投影精确为：

~~~text
v[j] = sum_g sx[g] * (a[j,g] * P[j,g] - c[j,g] * S[g])
~~~

任何后续拆分都必须保留 P 和 c*S 两项。忽略 Q4_K 的 offset correction 会产生系统性误差，而不是普通舍入噪声。

## 4. 基项 + 残差的三类公式

### 4.1 居中块拆分

对一个 32 值组，选择共享基值 A、C：

~~~text
x[i] = A + b[i]
q[i] = C + d[i]
~~~

点积恒等式为：

~~~text
sum_i x[i]q[i]
  = 32*A*C + A*sum_i d[i] + C*sum_i b[i] + sum_i b[i]d[i]
~~~

若 A、C 使用精确均值，使 sum(b)=sum(d)=0，则化为：

~~~text
sum_i x[i]q[i] = 32*A*C + b^T d
~~~

前一项只需块级聚合，后一项是唯一的细粒度残差点积。若使用整数 floor/round 基值，则保留两个显式的组和修正项即可维持精确性。

### 4.2 固定权重值表拆分

对固定权重码 q[i] 选择有限中心表 C[k]，并在冷启动时固定索引 h[i]：

~~~text
q[i] = C[h[i]] + r[i]
I[k] = { i | h[i] = k }
T[k] = sum_{i in I[k]} z[i]
R    = sum_i z[i]r[i]
P    = sum_k C[k]T[k] + R
~~~

C*T 是基项聚合，R 是锯齿残差。该表只编码固定权重，不是对激活做 VQ，也不要求运行时从输入码本查找权重。

### 4.3 层级基项

可继续按 32 值组、8 值子块和细粒度码分层：

~~~text
q[i] = b32 + b8[subblock(i)] + C[h[i]] + r[i]
P = b32*S32 + sum_t b8[t]*S8[t] + sum_k C[k]*T[k] + R
~~~

只要 R 完整保留，该式在整数域内仍是精确的。层级拆分的价值不在于减少算术，而在于把更多权重信息变成少量共享聚合，把必须搬到 GPU 的部分集中成连续 residual tile。

## 5. 共享低秩基底的推广

标量均值是常数向量上的 rank-1 特例。更一般地，对一个块选共享基底 U：

~~~text
x = U*alpha + b
q = U*gamma + d
U^T b = 0
U^T d = 0
~~~

则：

~~~text
x^T q = alpha^T gamma + b^T d
~~~

这给出了“主项占据小子空间、锯齿部分保留为残差”的严格条件。实际实验表明，普通 SVD 低秩在当前层上仍会留下较大的稠密残差，因此低秩只应在残差有真实结构、且静态/动态字节账本通过时采用。

## 6. 允许误差时的界

若残差使用 r_hat 代替完整 r，单个量化组的 pre-activation 误差为：

~~~text
e[j,g] = sx[g] * a[j,g] * sum_i z[g,i] * (r[j,g,i] - r_hat[j,g,i])
~~~

可用以下界做路由：

~~~text
|e[j,g]| <= |sx[g]a[j,g]| * ||z[g]||_1
             * max_i |r[j,g,i] - r_hat[j,g,i]|
~~~

合并所有组后，分别得到 delta_g、delta_u。最终 FFN 误差必须按原始非线性传播：

~~~text
delta_h = SiLU(g + delta_g) ⊙ (u + delta_u) - SiLU(g) ⊙ u
delta_y = W_down * delta_h
~~~

因此报告必须同时给出 gate/up pre-activation 误差和最终 FFN 输出误差，不能只报权重或残差的均方误差。超出预算时走更完整 residual 或精确回退。

## 7. 冷启动编译与运行时数据流

编译器对每个 layer、projection、output_tile、input_group 枚举有限候选：

~~~text
centered_radix / value_table / hierarchical
residual: exact signed / packed 2-bit / packed 4-bit / fallback
tile: 256 KiB / 1 MiB / 4 MiB / 16 MiB
~~~

选择目标不是单一压缩率，而是：

~~~text
J = residual_error
  + lambda_gpu * resident_bytes
  + lambda_h2d * dynamic_bytes
  + lambda_ops * extra_ops
  + lambda_cpu * routing_cost
~~~

运行时使用 pinned host ring buffer、异步 H2D 和双缓冲/三缓冲。逻辑页只负责目录与生命周期；实际 DMA 应合并为超级块，避免 4 KiB 小包带来的 API 和同步开销。每层可以独立选择：

~~~text
formula-only
formula + residual
full GPU FFN
exact fallback
~~~

手机集群只是 CPU 基项池的可选扩展。若手机按输出行分片，中心端关键路径可用：

~~~text
T_layer = max(T_gpu_residual,
              T_broadcast + T_phone_compute + T_base_return)
~~~

它首先缓解单机 CPU/L3 墙，不会自动消除中心 GPU residual 的 PCIe 下界。

按 Qwen3.8-27B 尺寸（hidden=5120、ffn=17408、64 层）、1 Gb/s 网络、12 GB/s PCIe 和 2048-row tile 的分析模拟，结果如下。这里的 phone base ready 已包含广播、手机计算和 base 回传；不是 Android 实测：

| 手机数 | 每手机 tile 表 | phone base ready | GPU residual critical | 单层关键路径 |
|---:|---:|---:|---:|---:|
| 1 | 1280 MiB | 3.78 ms | 5.57 ms | 5.57 ms |
| 2 | 640 MiB | 2.59 ms | 5.57 ms | 5.57 ms |
| 4 | 320 MiB | 1.98 ms | 5.57 ms | 5.57 ms |
| 8 | 160 MiB | 1.65 ms | 5.57 ms | 5.57 ms |
| 16 | 80 MiB | 1.49 ms | 5.57 ms | 5.57 ms |
| 32 | 40 MiB | 1.40 ms | 5.57 ms | 5.57 ms |
| 64 | 20 MiB | 1.35 ms | 5.57 ms | 5.57 ms |

该模拟说明手机池可以把 CPU 基项分支压到 GPU residual 分支之后，但在当前参数下不能继续降低 GPU 的 PCIe 下界。若把 residual 搬运或 down 策略优化到约 2.2 ms/layer，手机数量才可能重新影响单 token 临界路径。

## 8. 最高评价指标

每轮实验至少记录：

~~~text
GPU resident peak
H2D bytes/token/layer
GPU MAC/FLOP per H2D byte
copy active time
compute active time
copy/compute overlap
GPU wait/synchronization time
CPU routing and merge time
exact fallback rate
~~~

FLOPs 不减少并不构成失败；只要增加的算术换来了更高持续算子利用率和更低等待时间，方向仍然成立。反之，数学上完全等价但运行时仍需扫描同等大小稠密权重流的拆分，不满足本项目的带宽目标。

## 9. Qwen3.8-27B 适配判断

官方模型卡与配置见：

- Qwen3.8-27B 模型卡：https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-27B config.json：https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json

配置给出的关键尺寸为：

~~~text
hidden_size       = 5120
intermediate_size = 17408
num_hidden_layers = 64
hidden_act        = silu
~~~

官方 text_config 没有列出 MoE experts，可按单一路径 dense language tower 处理，并包含标准的 gate_proj、up_proj、down_proj 形态；其中 linear_attention/full_attention 的混合只影响整层调度，不改变单层 FFN 的代数边界。

按当前预算脚本的保守参数（2-bit 完整 residual、group=32、残差组 scale 4 bytes、有效 PCIe 12 GB/s）估算：

| 项目 | 单层/单 token 预算 |
|---|---:|
| 每个 gate/up/down 投影矩阵元素 | 89,128,960 |
| gate+up residual 包 | 63.75 MiB |
| gate+up base 返回 | 136 KiB |
| 单层 fp16 down 常驻 | 170 MiB |
| 双缓冲 full-chain workspace | 15.28 MiB |
| 单层 down 常驻 + 1 GiB reserve | 约 1.18 GiB |
| 全 64 层 down fp16 常驻 | 约 11.9 GiB，不适合 8 GiB 显卡 |
| gate/up + resident down H2D 下界 | 约 5.58 ms/layer |
| down Q4 流式 H2D 下界 | 约 9.76 ms/layer |
| down 也拆 residual 的 H2D 下界 | 约 8.37 ms/layer |

因此判断是：

1. 结构兼容：是。FFN 的 gate/up -> SwiGLU -> down 链路正好符合本报告的合并顺序。
2. 8 GiB 单卡直接常驻全模型：否。即使只看 fp16 down，全层常驻也超过显存；需要逐层驻留/流式分页。
3. 带宽交换有希望：有条件。63.75 MiB 的 gate/up residual 仍然很大，必须继续优化残差位宽、块复用、超级块合并和 down 放置；不能只凭“拆分”二字宣称已经解决 PCIe 瓶颈。
4. 现有 2B artifact 可复用：否。需要读取 Qwen3.8-27B 实际权重，按 5120/17408 的分块和量化元数据重新编译，并重新校准每层公式、误差阈值和回退率。

## 10. 已验证与未验证边界

已验证：

- Q4_K 主项/残差整数重构；
- layer 23 的 GPU residual、base+residual、SwiGLU、down 桥接；
- C++ 常驻 CPU 线程池和双缓冲异步 H2D；
- 27B-class 显存/H2D/计算密度预算；
- 旧手机输出行分片的网络临界路径模拟。

尚未验证：

- Qwen3.8-27B 真实 checkpoint 的编译 artifact；
- 8 GiB RTX 4070 上的 27B 端到端 tokens/s；
- 真实 Android 手机网络、热 throttling 和故障重路由；
- 多层误差累积、KV cache 和长上下文。

本报告适合作为公开实验协议和数学基线，不是“27B 已经在 8 GiB 显卡上跑通”的声明。
