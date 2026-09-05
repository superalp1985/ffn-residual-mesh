# FFN Residual Mesh

## A Base--Residual Decomposition for Bandwidth-Bound Inference

**Author:** Bingqin WANG  
**Version:** 0.1  
**Date:** 2026-09-05

## Abstract

现代 GPU 的峰值算力很高，但本地推理中的 FFN 经常不是算力不够，而是有效计算密度不够：矩阵单元在等待权重搬运、显存分页、PCIe/H2D 传输和同步。FFN Residual Mesh 的目标不是减少 FLOPs，而是重新安排计算和数据的位置：冷启动时完整扫描一次权重，把规律性强的部分编译成主机或 worker 可驻留的 base，把不规则部分保留为 residual，并在 GPU 上完成高密度计算和合并。

这构成一种资源等价交换：

```text
冷启动时间 + CPU/平板/手机计算 + 额外 GPU residual 算术
        <-> 更少动态权重搬运 + 更低显存压力 + 更少 GPU 等待
```

## 1. 从标准 FFN 开始

以 gated FFN 为例：

```text
g = W_gate x
u = W_up   x
h = SiLU(g) ⊙ u
y = W_down h
```

其中 `x` 是输入，`g` 和 `u` 是 gate/up 投影，`⊙` 是逐元素乘法。

对每一层、每一个投影，定义：

```text
W = B + R
```

`B` 是 base，适合在冷启动阶段预展开并长期放在 RAM、平板或手机内存；`R` 是 residual，适合组成连续 tile 后交给中心 GPU。

于是：

```text
g = Eval(B_gate, x) + Eval(R_gate, x)
u = Eval(B_up,   x) + Eval(R_up,   x)
h = SiLU(g) ⊙ u
y = W_down h
```

关键顺序是：**先合并 gate/up 的 pre-activation，再执行 SiLU 和门控乘法。**

## 2. 拆分为什么可以精确

最基本的恒等式是：

```text
(a + b)(c + d) = ac + ad + bc + bd
```

如果把输入和权重按块写成：

```text
x = x_base + x_residual
w = w_base + w_residual
```

那么一个点积必须保留四项：

```text
xᵀw
= x_baseᵀw_base
 + x_baseᵀw_residual
 + x_residualᵀw_base
 + x_residualᵀw_residual
```

项目中的 base 公式负责有限聚合，residual 负责不规则细节。只要交叉项没有被丢弃，拆分只是计算位置变化，不是数学近似。

## 3. 量化整数域中的有限聚合

对一个量化组，令权重码为 `q_i`，输入为 `x_i`。把固定权重码拆成有限中心值和残差：

```text
q_i = C[h_i] + r_i
```

其中 `C` 是有限中心表，`h_i` 是冷启动时固定的索引，`r_i` 是完整残差。定义：

```text
T_k = sum(x_i),  for all i with h_i = k
R   = sum(x_i r_i)
```

则点积为：

```text
sum(x_i q_i)
= sum_k C[k] T_k + R
```

第一项只需要读取有限状态的部分和，第二项是 residual 点积。若量化解码为：

```text
w_i = scale * (q_i - zero_point)
```

则：

```text
sum(x_i w_i)
= scale * (sum_k C[k] T_k
           - zero_point * sum_i x_i
           + R)
```

`zero_point` 修正项不能省略。该式说明了本算法的精确边界：在选定的整数/量化表示内，完整 residual 可以严格恢复原始投影；只有裁剪、舍弃 residual 或改变非线性顺序时，才会产生算法误差。

## 4. 为什么不能跨过 SwiGLU 合并

正确路径是：

```text
g = g_base + g_residual
u = u_base + u_residual
h = SiLU(g) ⊙ u
```

不能写成：

```text
SiLU(g_base) ⊙ u_base
+ SiLU(g_residual) ⊙ u_residual
```

因为一般情况下：

```text
SiLU(a + b) != SiLU(a) + SiLU(b)
```

所以 base/residual 的合并点必须位于 gate/up 的 pre-activation 空间。`down` 投影在 `h` 得到以后执行，可以选择整层驻留、分块流式或再次拆成 base/residual。

## 5. 资源等价交换模型

对一层 FFN，中心 GPU 和 worker 的临界路径可以近似写成：

```text
T_layer = max(T_gpu_residual,
              T_base + T_transport)
```

其中：

- `T_gpu_residual`：残差 H2D、GPU residual 算子、合并和同步；
- `T_base`：CPU、平板或手机生成 base 的时间；
- `T_transport`：descriptor 广播和 base tile 回传时间。

本项目追踪的不是“压缩率”本身，而是：

```text
effective_compute_density = useful_GPU_ops / transferred_byte
```

只要 `T_base + T_transport` 能隐藏在 GPU 分支后面，额外的 base 计算就不会增加关键路径；即使 FLOPs 增加，也可能换来更低的 GPU 等待和更高的持续算子利用率。

## 6. 手机和平板 worker

worker 不需要替代 GPU，也不需要保存完整模型。它只保存编译后的 base artifact，并按输出行或 tile 分片：

```text
中心端发送 descriptor
    -> worker 计算 base tile
    -> 中心 GPU 计算 residual tile
    -> gate/up 合并
    -> SwiGLU、down、attention 继续在 GPU
```

对多个设备，中心端可以并发调度：

```text
T_phone = T_broadcast + T_worker + T_return
T_critical = max(T_gpu, T_phone)
```

worker 超时、checksum 失败、温度过高或误差预算不满足时，直接回退中心 GPU 精确 FFN。这样集群是可选后端，不会破坏原有 ComfyUI 工作流。

## 7. 应用前景

### MiniMax H3 + ComfyUI

H3 的视频 FFN 有很长的 packed sequence，适合验证“中心 GPU 做 residual，外部 worker 做 base”的调度边界。当前模拟显示：精确 gate/up 回传在 1 Gb/s 下会被网络主导，而 10 GbE 或高吞吐 USB/平板链路有机会把 worker 分支隐藏在 GPU 计算之后。TeaCache 只在 real step 调用 worker，可进一步降低集群流量。

### 27B-class dense 模型

对 8 GiB 显卡，最现实的路线不是把所有权重都塞进 VRAM，而是：

```text
base artifact：主机 RAM / 平板 / 手机
residual tile：GPU 显存
SwiGLU/down：GPU 计算
```

这为 Qwen3.8-27B-class 等更大 dense 模型提供了一个可测量的显存和带宽交换方向，但仍需真实 checkpoint 编译和端到端验证。

### 边缘设备和旧硬件

旧手机、平板、迷你主机和 NAS 可以作为分布式 base worker。它们贡献的是内存容量、串行/向量计算和并行设备数量；中心 GPU 仍负责最适合 GPU 的 residual 和非线性算子。

## 8. 精确与近似的边界

精确路径：

- residual 完整保留；
- 所有 scale、offset、进位和交叉项都纳入合并；
- gate/up 在非线性前合并；
- 结果在选定量化整数域内与原始投影一致。

近似路径：

- residual 被量化、裁剪或低秩化；
- base 公式只覆盖部分状态；
- 需要单独报告 pre-activation 和最终 FFN 输出误差；
- 超出误差预算就回退精确 GPU 路径。

因此，“精确拆分”与“有损压缩”必须分开报告。前者改变的是数据布局和执行位置，后者才改变函数本身。

## 结语

FFN Residual Mesh 的核心不是让计算凭空消失，而是把计算放到更合适的位置：让主机和设备内存承载规律性强的 FFN base，让 GPU 集中处理 residual、合并和非线性。对于显存和带宽受限、但 GPU 计算单元没有被充分喂饱的系统，这是一种可以用字节数、等待时间、重叠率和精度共同验证的资源等价交换。

## Related Files

- `docs/math_principles_report.md`
- `docs/weight_code_split_spec.md`
- `docs/comfyui_phone_cluster_design.md`
- `scripts/simulate_comfyui_phone_ffn.py`

**Bingqin WANG**
