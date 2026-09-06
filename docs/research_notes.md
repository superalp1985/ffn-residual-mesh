# 调研与设计记录

日期：2026-09-02

## 2026-09-03 最高优化原则

本项目不以减少 FLOPs 为目标。允许通过解码、张量收缩、交叉项和残差重构增加总计算量，换取：

1. GPU 常驻字节下降。
2. 每 token H2D 动态搬运字节下降。
3. 每个 H2D 字节触发的 GPU 运算增加。
4. GPU 等待 PCIe、显存数据和同步的时间下降。

主机 RAM 是冷启动算子仓库，允许预展开并长期驻留；预展开不是把完整 FFN 矩阵乘移到 CPU：

```text
E_l = Compile(Wg_l, Wu_l, Wd_l)
q_l = Encode(x_l)
y_l = EvalGPU(E_l, q_l)
```

正常路径不得完整扫描原始 `Wg/Wu/Wd`。原始权重只允许在冷启动编译和精确回退时访问。CPU 完整计算基础输出的现有实验只作为正确性和流量基线，不再作为最终架构。

对多项式 SwiGLU，冷启动可把原始权重组合为线性项和高阶张量项；高阶项必须做跨通道 CP/Tucker/Tensor-Train 分解，不能用原神经元通道裁剪代替张量压缩。

## 结论

方向可行，但研究对象应从“权重拆分”提升到“输入条件算子展开”。权重域中的简单残差不能跨过非线性直接相加：

```text
activation((W + R)x) != activation(Wx) + activation(Rx)
```

因此必须把非线性交叉项纳入残差校正或精确回退。

## 预展开的三层含义

1. **布局展开**：权重分块、量化、重排和对齐，优化 CPU cache、SIMD 和 GPU tile。
2. **状态展开**：根据校准激活建立状态中心 `c[k]` 和基础输出 `M[k] = F(c[k])`。
3. **残差展开**：为状态区域拟合共享低秩、分块稀疏或 LUT 残差算子。

## 分页式块搬运

Windows 的页式思想可借鉴，但不能直接把 4KB 页面作为 PCIe 搬运单位。建议：

- 逻辑页负责路由和生命周期管理；
- 超级块负责连续搬运；
- 请求按层、地址和热度排序后合并；
- 使用 pinned host memory，配合双缓冲和异步复制；
- 维护精确回退权重，避免表缓存失效时阻塞模型运行。

一个块目录可以包含：

```text
block_id, layer_id, offset, bytes, format, hotness, checksum
```

块大小必须实测。太小会放大 DMA、API 和同步开销，太大则会搬运大量未使用数据。

## 算子适配

不同层不强行使用同一种公式。离线编译器根据误差和硬件成本为每层选择：

```text
lookup-only
lookup + low-rank residual
block LUT for W1
full GPU FFN
```

运行时还要有 OOD 误差估计。`||x-c[k]||`、残差预测幅度和历史回退率都可以作为第一版信号。

## 研究风险

- 普通 FFN 的输出依赖上下文，静态表不可能覆盖所有状态；纯查表只能作为基线。
- 每个状态中心都存一套完整残差基底会导致表空间爆炸，优先测试共享基底。
- 小 batch decode 中，GPU 可能受 kernel launch 和同步影响；prefill 与多请求 batch 的结果可能不同。
- 长上下文场景中，KV cache 仍是独立的显存问题。

## 当前模型特殊性

Qwen3.5 使用 Gated DeltaNet 线性注意力与 Gated Attention 的混合栈，官方文档描述为约 3:1 的混合比例。当前 GGUF 的 `qwen35` 张量同时包含 `ssm_*`、`attn_*` 和 `ffn_*`；所以 FFN 仍可单独实验，但端到端数据必须按 block 类型分层统计。

## 相关工作定位

- FFN 可被视作一种 key-value memory，支持从输出贡献角度做分解。
- PowerInfer 说明 CPU/GPU 协同处理热、冷神经元有现实收益。
- SoLA 说明 FFN 激活能量存在长尾，可尝试保留高贡献部分、压缩低贡献部分。
- LUT-LLM、T-MAC 和 FLUTE 说明离线查表、位打包和权重布局会影响实际吞吐。
- MemoryLLM 说明只有经过特殊训练的 context-free FFN 才适合直接做 token 级静态查表。

## 2026-09-02 二次调研与实验决策

本轮先查了 LUT-LLM、FLUTE 和 TARDIS，再决定实验方向：

- LUT-LLM 已把向量量化、中心搜索、二维表和空间/时间缓存作为完整硬件路径；本项目不重复实现通用 LUT-GEMM，而把重点放在 CPU 分类、GPU 残差回退和 Windows 式超级块搬运的组合。
- FLUTE 的离线重排、查表复制和向量化说明“预展开”必须包含布局与缓存副本，而不是只保存数学公式。
- TARDIS 与本项目的局部线性残差最接近：常见输入区间走线性近似，离群输入在线回退精确 FFN。它主要验证了“分段线性 + 回退”的方向，但 Qwen3.5 使用并行 SiLU 门控，误差需要单独实测。

这三项工作把本项目的差异收敛为：不训练新模型，针对现成 Qwen3.5 GGUF 做离线状态表；运行时用 CPU 路由和块目录决定是否只搬运小残差；每层独立选择 lookup、低秩残差或完整 FFN。

## 参考来源

- Geva et al., Transformer Feed-Forward Layers Are Key-Value Memories.
- PowerInfer, Fast Large Language Model Serving with a Consumer-grade GPU.
- SoLA, Training Smaller Language Models by Consolidating Layers and Removing Redundancy.
- LUT-LLM, Neural Network Inference with Lookup Tables.
- T-MAC, CPU-Friendly Low-Bit LLM Inference with Table Lookup.
- FLUTE, Flexible Lookup Table Engine for Accelerating DNN Inference.
- MemoryLLM, Towards Self-Updatable Large Language Models.

## 2026-09-02 三次调研与实验决策

本轮在实现输入侧低秩残差前再次核对公开路线：

- PowerInfer（arXiv:2312.12456）采用热神经元 GPU 常驻、冷神经元 CPU 计算，并强调激活局部性和自适应预测器；这支持“CPU 分类 + GPU 小子集”的系统形态，但不证明 dense FFN 可被静态表完整替换。
- FlexGen（arXiv:2303.06865）将 GPU、CPU、磁盘视为统一资源并搜索张量存取计划；本项目的超级块、预取和双缓冲应沿这个方向建模，而不能只看理论字节数。
- T-MAC（arXiv:2407.00088）与 LUT-NN（arXiv:2302.03213）分别验证了低比特矩阵乘的 CPU 查表，以及“中心 + 预计算输出”的通用算子查表；两者都依赖离线布局/中心学习，且 LUT-NN 的端到端结果来自训练适配模型，不能直接外推到未再训练的 Qwen3.5。
- llama.cpp 当前提供 `--n-cpu-ffn`、`--override-tensor` 和 eval callback；后续原型可先用现有 CPU/GPU 张量放置和 callback 做注入，不必立即重写调度器。

本轮实验决策：先实现输入 `x` 的局部低秩模型和 OOD 回退，优先验证 layer 23；layer 22 作为负对照。只有在混合路径的模型级 KL 和实际 callback 时间可接受时，才进入图执行器改造。

## 2026-09-02 计划收敛

用户确认降低目标难度：当前不追求全模型统一公式，也不追求复杂的跨层合并。研究主线改为：

```text
每层权重/布局预展开到 RAM
        -> CPU 路由与系数生成
        -> GPU 计算少量残差
        -> 本层内轻量 base + residual 合并
        -> 超阈值时整层精确回退
```

“分层合并”只要求本层输出正确进入下一层，不要求建立额外的跨层状态协议；KV cache 以及后续长上下文问题后置。这样首要优化目标变成 H2D 字节、显存峰值、GPU 空转和 CPU 路由开销之间的平衡。

当前优先级：

1. layer 23 单层真实 callback/replay。
2. layer 22 负对照，确认按层选择必要性。
3. 连续 token 的超级块合并和 pinned ring buffer。
4. 只扩展到少量稳定层，验证串联 KL 和吞吐。

## 2026-09-03 预展开与查表的区分

本轮确认一个重要原则：**预展开到内存不等于必须查表**。

“预展开”描述的是离线产物和布局：权重块、基础输出、残差基底、分段参数、量化副本可以提前生成并放在 RAM 中；“查表”只是运行时根据输入选择这些产物的一种方式。可选运行路径包括：

1. 直接按层/块地址读取预展开数据，再做轻量差值。
2. 用索引或中心查找选择预展开块，再做残差计算。
3. 直接读取分段公式参数，计算低阶非线性差值。
4. 对无法稳定近似的输入整层精确回退。

因此后续实验不能默认 LUT 一定优于直接读取。应分别统计分类、随机内存访问、顺序块读取和残差合并的成本，比较：

```text
direct block read + residual
indexed lookup + residual
piecewise formula + residual
```

如果输入路由已经由层号、块号或连续 token 顺序确定，直接取预展开块可能比查表更快，也更容易利用 CPU cache 和超级块搬运。

## 2026-09-03 算术等价拆分的边界

用户进一步明确：主线应是固定权重的“臃肿基项 + 锯齿残差”算术拆分，而不是把激活压入 VQ 码本。每层/每块可有自己的非线性、分段或表驱动公式；不要求推导一个跨模型通用公式。

该路线不以传统机器学习的“过拟合”作为主要限制。真正的验证项是：量化整数域的等价性、保留输入的覆盖、异常输入回退率、公式表大小和运行时字节账本。

对定点表示：

```text
x = Bx*x_hi + x_lo
w = Bw*w_hi + w_lo

x*w = Bx*Bw*(x_hi*w_hi)
    + Bx*(x_hi*w_lo)
    + Bw*(x_lo*w_hi)
    +     (x_lo*w_lo)
```

矩阵投影把四项分别累加。它可用不同硬件、不同块公式执行，随后用尺度/进位规则合并；只要四项都保留，拆分在该整数域中精确。原始 SiLU 和 down 投影应在 `g/u` 合并后保持不变。

重要限制：高位项本身若仍是无结构的稠密矩阵，则它仍然需要每 token 访问同等信息量。真正可能减少搬运的候选是把高位项编译成块共享基项、位平面算子、低秩/分层基项或其他可由少量输入聚合值生成的结构；残差再以较高计算密度处理。该限制是信息访问约束，不是要求所有拆分公式线性。

因此，下一组实验的顺序调整为：

1. 对 `Wg/Wu` 做逐层、逐块 `H + R` 的精确重构测试。
2. 用定点位拆分完整核对四个交叉项与合并误差。
3. 分别测无结构高位项、块共享高位项和残差项的静态/动态字节，淘汰“数学等价但没有减少读取”的公式。
4. 只把通过字节账本的公式接入 GPU SiLU/down 路径。

## 2026-09-06 设备侧链路调研与实现

本轮先核对了 PyTorch CUDA stream/event 与 CUDA Graph 的官方语义，再修改设备
激活路径。`Stream.wait_event` 适合表达跨 stream 依赖；CUDA Graph 适合固定形状
工作链的重复提交，但不减少 FFN 算术量。由此得到一个更窄、可验证的优化：

```text
GPU activation 已经存在
    -> 同一消费 stream 计算 group_sums
    -> fused residual + resident base + SwiGLU
    -> resident down
    -> 下一层直接消费
```

对已经在 GPU 上的 activation，额外 D2D copy 或独立 base stream 没有可隐藏的
H2D 工作，反而会增加 event 依赖。因此 `run_device()` 现在直接使用调用者 tensor，
并提供 `synchronize=False` 让多层调用在一个 stream 上连续排队。异步模式只返回
completion event，不允许在 GPU 完成前读取输出。

真实 Qwen3.8-27B layer 3/21 的两层链测得约 17.63 ms GPU span、约 2.77 ms CPU
enqueue；两层 activation/base 的动态 H2D/D2D 账本均为零。这是减少层间主机往返和
提交空隙的证据，不是完整模型 token/s 结论。下一阶段应继续减少 Python/event
开销，并用系统级 profiler 测量 kernel gap，而不是只看单层 wall time。

## 2026-09-03 资源导向的共享基底扫描

本轮不把共享基底当成必须采用的固定答案，而是把它放入“GPU 计算效率换带宽/显存”的候选集合。对 layer 23 的 gate/up 做了共享右基底与各自右基底的数学扫描，秩为 8、16、32、64、128、256，并对残差做 2/3/4/8-bit 对称分组量化。

主要结果：

- 共享基底与独立基底在相同秩下的 gate/up 投影误差都很低（约 `1e-4`，未量化残差），但这只是分解恒等式的数值核对。
- 权重残差 Frobenius 比例在 rank=8 时仍约 `96%`，rank=256 时仍约 `81%`（共享基底）；说明普通 SVD 低秩并没有把大部分权重信息变成可省搬运的“基项”。
- 残差 4-bit 后，rank=256 的 FFN 留出误差约 `3.3%`（相对 float 权重教师）；2-bit 约 `19.5%`。但此时残差仍是稠密存储，尚未形成带宽收益。
- 共享基底在 rank=256 比独立基底少约 `1 MiB` fp16 基底/系数静态空间；独立基底残差略小，但差异不足以改变结论。

因此后续不再机械追求“同一个 U”。真正的筛选目标是：共享/独立/分区基底都可选，但残差必须进一步变成短码、位平面、共享模板或其他规则结构；否则只是把一份稠密权重换成另一份稠密权重。

## 2026-09-06 单 kernel IQ4_NL down 与 DMA 边界复核

本轮先复核 CUDA stream/event、异步拷贝和 CUDA Graph 的语义，再处理层内
人为的串行边界。原 `DirectIQ4NLProjection.launch()` 是：

```text
多个 K 分块 IQ4_NL kernel -> partial 矩阵写回 -> torch.sum 归约
```

现在改成 `_fused_iq4nl` 单 kernel：每个 row program 在寄存器中按 K tile
累加，保留原 IQ4_NL 16-bit half scale 解码，最后一次性写出 output。该改动
通过真实 IQ4_NL fixture、完整 CUDA 测试和 CUDA Graph down 捕获校验。

真实 Qwen3.8-27B layer 3（RTX 4070 Laptop，PyTorch 2.9.1+cu130）：

```text
group_sum                         median 0.0225 ms
fused gate/up/base/residual       median 0.7808 ms
fused IQ4_NL down                 median 0.4905 ms
整条 GPU event span                约 1.94 ms（样本受频率抖动影响）
```

layer 21 的对应中位数为 `0.0192 ms / 0.8457 ms / 0.4905 ms`。因此
`0.3~0.5 ms` 目前只在单独 down GEMV 上达到，不能代表完整
`gate + up + SwiGLU + down`。

设备激活两层链的重新测量：

```text
主机逐层同步 GPU span       median 3.31 ms
链尾一次同步 GPU span       median 3.14 ms
显式异步同 stream           median 3.16 ms
CPU enqueue                 median 0.38 ms
每层 activation/base H2D    0 B
```

这组数据不能支持“GPU 没有通过硬件 DMA”的判断：常驻设备路径根本没有运行时
activation/base H2D，因而没有需要 DMA 的传输。层间仍存在
`layer N output -> layer N+1 input` 的数学依赖；异步 stream 只能消除主机提交
等待，不能把依赖的两个 GEMV 重叠。需要进一步突破的是 int4 GEMV 的 kernel
效率、量化解码与点积融合，以及换入路径的 pinned-memory copy/compute 双缓冲，
而不是给常驻路径额外添加 D2D 或 event。

## 2026-09-03 激活敏感度选块

简单按 `||W_b-W_hat_b||_F` 选低秩块会误判 gate/up 的实际影响。更合理的拆分评分是：

```text
score_b = E_x || delta_y,b(x) || / ||y(x)||
```

gate/up 块要经过 SwiGLU 门控传播，down 块直接在 hidden 输出空间评估。只有在真实激活下低影响、且因子存储小于精确 Q4 块时，才允许拆分。layer 23 的初步结果显示，激活敏感度排序比 SVD 残差排序更有效，但当前可接受节省仍只有个位数到十几个百分点。

## 2026-09-03 非查表代数拆分实验

本轮按“增加算力、减少动态非线性和权重搬运压力”的目标，测试了三类公式：

1. 全局输入低维多项式特征：`phi=[1,z,z^2,...]`。
2. gate/up 低维双线性交叉特征：`phi=[1,z_g,z_u,z_g\otimes z_u]`。
3. 按中间神经元的 SwiGLU 多项式：

   ```text
   h_j = u_j * p_j((g_j - mu_j) / sigma_j)
   y = sum_b W_down,b @ h_b
   ```

结果显示，全局低维特征在 layer 23 的最佳留出误差仍约 0.25--0.30；双线性交叉项在低秩时容易过拟合，rank=24 以后留出误差爆炸。它们暂不作为主公式。

按神经元拟合的多项式更稳定：

- layer 23：degree=4，留出输出 rel-L2 约 0.0283，激活 rel-L2 约 0.0394；degree=3 约 0.0371。
- layer 22：degree=3，留出输出 rel-L2 约 0.0741，激活 rel-L2 约 0.0644。
- 每层只需保存每个中间神经元的均值、尺度和少量多项式系数；degree=4 时约 86 KiB(fp16) 的系数产物。

这证明“非线性可以拆成预展开系数 + 运行时乘加”，但还没有证明权重搬运一定下降：gate/up/down 的矩阵乘仍需权重。下一步必须把该公式与分块权重拆分结合，单独测动态 H2D 字节和块复用率。

## 2026-09-03 残差系数目标与稀疏路由

按神经元多项式基础项接入低秩输出残差后，残差拟合目标需要区分两种语义：

```text
r_exact = W_down @ h_exact - y_base
r_capture = y_captured - y_base
```

`r_exact` 更接近数学重放，`r_capture` 更接近量化模型实际输出。layer 23 上，degree=4、输入秩 128、输出秩 64 时，拟合 `r_capture` 的留出误差约 0.0267，优于拟合 `r_exact` 的约 0.0275；layer 22 仍约 0.0766，说明目标函数不能替代分层适配。

残差系数可以继续做 CPU top-k：

```text
I_k = TopK(|alpha(x)|)
y_hat = y_base + mu_r + sum_{i in I_k} alpha_i U_i
```

layer 23 的 `output_rank=64` 上，top-k=4（16 B fp16 值 + 8 B bitmask）与完整系数误差接近，证明“系数稀疏化”比单纯提高输出秩更划算。实际部署必须计入 bitmask/索引和 GPU gather 成本。

由于 SVD 输出基底正交，省略系数的二范数可以作为第一版路由信号：

```text
tail(x) = ||alpha(x) - TopK(alpha(x))||_2
```

用校准分位数设置阈值后，layer 23 可在约 97--99% token 走近似路径时保持约 2.66--2.68% 的留出误差代理，并把期望 down 权重搬运降到约 0.08--0.23 MiB/token。layer 22 即使全近似也约 7.7%，必须回退。

以上 H2D 数字是流量代理，不代表 llama.cpp 已实现；下一阶段必须用连续 token、超级块和 pinned ring buffer 测真实 DMA、同步和 kernel 时间。

## 2026-09-03 Chebyshev 基底扫描

monomial degree=4 在部分层上存在区间外数值放大。将标准化 gate 限制到 `[-B,B]`，并用 Chebyshev 递推拟合后，layer 23 和 layer 22 的留出误差均明显下降，但最优 `B` 不同，因此参数必须逐层记录：

```text
layer 23: degree=5, B=5
layer 22: degree=4, B=6
```

这不是对所有层都适用的全局公式，而是“每层选择算子”的证据。Chebyshev 只改变基础非线性展开，不改变残差、top-k 或精确回退协议。

## 2026-09-03 超级块与延迟预算

对当前约 `4 KiB/token` 的近似包做分页模拟后，发现 DMA 块大小必须和 token 窗口绑定：

- batch=1 强制 256 KiB 会把每 token 流量放大到 256 KiB；
- 16--64 token 连续窗口时，64 KiB 超级块更合适；
- 256 KiB 更适合 prefill 或更长窗口，不能作为 decode 固定值。

因此建议把传输层拆成两条队列：近似输出走顺序 pinned ring buffer，fallback 权重走按热度缓存的页/超级块队列。两者的粒度、生命周期和回收策略不同，不能共用一个“分页大小”。

## 2026-09-03 分层覆盖率初测

在 layer 0、10、18、22、23 上用相同 Chebyshev 搜索范围做初筛，得到明显不同的最优误差：layer 10 约 3.3%、layer 23 约 2.3%，layer 22 约 5.0%，而 layer 0/18 约 9.1%/7.5%。因此近似算子必须由离线 manifest 按 `layer_id` 选择，不能把末层参数复制到全模型。

layer 10 的完整残差链路约 3.2%，说明可用层不只末层；layer 0/18 当前应直接精确回退。后续系统原型先选 layer 10、22、23，其他层保持精确路径作为对照。

## 2026-09-03 CPU 成本核对

本机 NumPy 单 token 微基准显示，残差系数投影和 top-k 选择只有约 11--12 μs，而三次 FFN 投影约 2.2--2.7 ms。Chebyshev 基础项比精确 SiLU 路径增加几百微秒，说明新增路由算术本身不会吞掉主要收益；但这不是端到端加速证据，GPU merge、H2D 和同步仍需实测。

## 2026-09-03 完整残差与冷启动扫描边界

残差默认不做有损压缩。`q = 4*q_hi + q_lo` 中的 `q_lo` 是天然 2-bit 的完整低位残差；把它 bit-pack 只改变搬运字节，不丢信息。只有裁剪、舍弃、再次量化或省略 `q_lo` 才属于残差近似，相关误差必须单独报告。

冷启动允许完整扫描原始 Q4_K 权重一次，用于生成主项公式、部分和、有限状态表和精确残差 tile。运行时禁止再次扫描原始权重，也禁止顺序扫描等价的展开主项流；否则只是换了存储位置，没有解决运行时主项访问问题。

新增 `estimate_exact_state_table_ledger.py` 后，layer 23 的单个 6144x2048 投影得到以下账本（表项按 fp16 输出向量计）：

- 2-bit 激活状态、block=4：状态表约 1.5 GiB，运行时表读取约 6 MiB/token；完整 2-bit 残差包（含 Q4_K scale/min 元数据）约 3.75 MiB/投影。
- 3-bit 激活状态、block=4：状态表约 24 GiB，运行时表读取仍约 6 MiB/token。
- block=1 即使只保留 2-bit 状态，运行时读取仍等于 fp16 稠密矩阵，不能减少主项访问。

这说明完整残差在信息论和算术上没有问题，真正的难点是主项有限状态表的指数规模。后续只保留能够证明“有限表访问低于稠密主项流”的层/块布局；状态离散化留下的 `H*epsilon_x` 必须作为独立激活余项计算、界定或触发精确回退，不能误称为权重残差误差。

## 2026-09-04 目标指标修正：优先消除 GPU 等待

显存峰值不再要求相对原始路径下降。允许用更大的残差窗口、预取页和双/三缓冲换取 GPU 连续计算，只要峰值不超过设备容量。优化目标按以下顺序排列：

1. GPU copy/sync 等待下降；
2. H2D 与残差 kernel 的重叠上升；
3. GPU 持续算子利用率上升；
4. H2D 字节/token 下降；
5. 显存峰值只需满足不 OOM。

因此后续不能因为 resident bytes 没有下降就淘汰方案；必须看预取后 GPU 是否还在等待数据。残差完整保留仍是默认，显存中可以多放几个残差超级块来换取流水线连续性。

## 2026-09-04 CUDA 残差流水线与 CPU 主项并行

第 36 轮把真实的 QLO2 artifact 接入 CUDA runner：每个 tile 将 row-packed 2-bit code 与 alpha 合成单个 pinned H2D 包，copy stream 与 compute stream 通过 event 双缓冲连接。layer 23 在 1024-row tile 上，gate/up 合计的串行 copy+kernel 约 1.52 ms，双缓冲约 1.13 ms；残差 kernel 本身只有约 0.1 ms，因此不能把 residual 算力误认为可以单独覆盖全部 H2D。

第 37 轮把 block=4 radix partial-sum table 接入同一 runner。CPU 端按输出行使用常驻线程池，GPU 端继续处理完整 q_lo residual。8--12 个 CPU 线程时，gate/up 两投影的串行组件约 3.5--3.7 ms，并行墙钟约 2.3 ms，约 1.5x 改善。线程数继续增加没有稳定收益，说明 CPU 主项已接近内存带宽/调度拐点。

第 38 轮比较表大小：block=4 每投影 768 MiB，block=2 每投影 96 MiB。相同 1024-row residual 和 8 线程下，并行 pair 分别约 2.40 ms 与 3.21 ms。block=2 仍可用，是低 RAM 后备；block=4 是性能优先格式。格式应按层和主机资源选择，而不是全局固定。

当前结论：分层“CPU 预展开主项 + GPU 完整残差 + 轻量合并”已经有可重复的层级证据。尚未完成的硬门槛是集成 runner 的 `base + residual` 数值校验、SwiGLU 合并和 down 投影；在这些完成前不宣称端到端模型加速。

## 2026-09-04 GPU 计算密度桥接

第 40 轮完成了首个完整 GPU 链：GPU 接收 gate/up 的精确 2-bit residual 包，随后执行 `base + residual -> SwiGLU -> resident down`。此前 Triton alpha 切片保留了整包行跨度，但内核按紧凑矩阵寻址，导致第二行以后越界并出现 NaN；改为显式传入整包行 stride 和 alpha 偏移后，残差、合并、SwiGLU 均恢复到约 `1e-7` 相对 L2，fp16 down 输出约 `2.9e-4`。

layer 23 的总 residual H2D 仍为 9 MiB。block=2 的 20 次采样中，residual-only GPU 计算约 0.257 ms，接入 SwiGLU 和 resident down 后约 0.364 ms，计算段约 1.41x；block=4 的对照约 1.42x。这个结果支持“用额外 GPU 算子延长一次 residual 搬运后的工作段”这一方向，但当前仍是单 token、默认流的密度桥接，不等于端到端加速或硬件 occupancy 证明。

下一轮把这条链接入 CPU 主项 + GPU residual 双缓冲 runner，直接测 copy/compute overlap、critical path 和 GPU 等待区间。

## 2026-09-04 CPU 主项结果接入完整 FFN tile 链

第 41 轮完成了完整 tile runner：CPU 端使用预展开 radix 主项结果，GPU 端接收 gate/up 的 9 MiB 精确 2-bit residual 包和每 tile 8 KiB 的 base 辅助向量；每个 tile 在同一依赖链上执行 residual reduction、`base + residual`、SwiGLU 和 down partial，最后归约 6 个 tile。

block=2、1024-row tile、10 次采样中，串行 critical path 约 1.80 ms，双缓冲约 1.22 ms，约 1.47x；完整 down 输出相对 fp32 CPU 参考约 `1.18e-4`，所有 tile 的 SwiGLU 误差约 `1e-7`。这证明了“主项在内存预展开、残差进显存、后续算子延长 GPU 工作段”的完整算子链可以正确运行。

本轮同时修复了两个真实流水线问题：H2D copy 必须在 copy stream 上执行，且 down tile 必须加上全局列偏移。下一轮做 tile 大小 sweep，并把 CPU 表评估本身接入同一 measured loop，观察 CPU 生产与 GPU residual 是否能继续重叠。

## 2026-09-04 tile 大小 sweep

第 42 轮在清洁计时区间内比较了 512、1024、2048、6144 行 tile。残差 H2D 始终为 9 MiB；512 行因为 12 个 tile 的 launch/event 开销过高，双缓冲只有 1.09x；6144 行没有可重叠的多 tile，约 0.98x；1024 行约 1.36x；2048 行 critical path 最低，约 `0.95 ms`，双缓冲约 1.39x。因此当前 RTX 4070 候选默认取 2048 行，1024 行作为更细粒度 fallback。

所有 tile 大小的完整 down 输出误差都约 `1.18e-4`，说明该选择只影响调度，不改变拆分/合并公式。下一轮把 CPU radix evaluator 从预展开结果加载改成同一 measured loop 的生产端，直接测 CPU 生产与 GPU tile 是否能形成持续 overlap。

## 2026-09-04 Python 表生产的边界

第 43 轮把运行时 base 生产切换为 Python 线程池逐 tile 读取 radix partial-sum table。block=2、2048-row、8 线程下，base 生产约 300 ms，而 GPU copy/compute 仍只有约 1 ms；完整输出误差保持 `1.18e-4`。因此问题是 Python 小块循环和调度开销，不是拆分公式或合并精度。

Python table mode 只保留为正确性 oracle，不能作为目标运行时性能结论。目标实现必须复用已有 C++ 常驻线程池 evaluator，再把 CPU base tile 与 QLO2 residual 一起接入 GPU full FFN 双缓冲链。

## 2026-09-05 super-tile 基项/残差融合与 GPU 气泡

第 59 轮针对 H2D、tile、kernel 之间的空隙做了两步改动：

1. `base_on_gpu` 路径把每层的 gate/up coefficient 冷启动常驻显存；运行时只上传
   `160` 个 fp32 分组和，约 `640 B/token`，不再上传 `17408 * 2` 个 fp32 基项，
   后者约 `136 KiB/token`。
2. 当整层 residual 已经常驻、且 tile_rows 等于整层行数时，使用一个 super-tile
   kernel 同时完成：

   ```text
   Q4 residual dot + coefficient · group_sums + gate/up merge + SwiGLU
   ```

   这样去掉了 residual 中间向量写回、单独 base GEMV、以及第二次 merge launch。
   多 tile 或换入层仍保留原来的粗粒度 tile 路径，不强行融合。

fixture 和真实 Qwen3.8-27B 第 3、21 层均通过数值校验。真实层 warm A/B（RTX 4070
Laptop，单 token，整层 resident，未包含 attention）：

```text
layer 3:  CPU-base 约 1.51 ms，GPU fused super-tile 约 1.41 ms
layer 21: CPU-base 约 1.89 ms，GPU fused super-tile 约 1.33 ms
```

不同运行会有抖动，不能据此宣称模型端到端加速；但它证明了“把残差喂成一块密集
super-tile，并在同一个 kernel 内完成基项、残差和非线性合并”比细碎的
`residual -> base GEMV -> merge` 更接近目标调度形态。第 3 层收益较小，说明 kernel
本身已接近瓶颈；第 21 层收益更明显，说明原 CPU base 生产/上传确实暴露了关键路径。

当前实现的设备指标：

- 动态 base H2D：`1280 B`（160 个 fp64 分组和）旧实现；新 fused 路径已降为
  `640 B` fp32 分组和；
- coefficient resident：gate/up 合计约 `22.3 MiB`（fp32）；
- resident residual weight H2D：warm run 为 `0`；
- 误差：fixture 与真实 artifact 的 gate/up 最大绝对误差保持在 `1e-6` 量级；
- GPU 计算密度仍需用 Nsight Compute/Systems 读取 SM active、tensor/FP pipe active、
  memcpy overlap 和 kernel gap，墙钟时间本身不能替代 occupancy 证据。

调度结论：常驻层采用整层 super-tile；换入层采用 1024--4096 行的粗 tile + 双缓冲；
不要把 decode 的单 token 传输强行放大成 64/256 KiB 固定页。下一轮优先接 CUDA Graph
固定形状回放和 Nsight 指标，再决定是否保留独立 base stream。

## 2026-09-05 CUDA Graph 固定形状回放

第 60 轮把常驻整层路径接入可选 CUDA Graph。图内固定：

```text
激活 H2D + group_sums H2D
-> fused base/residual/SwiGLU
-> 可选 resident down
```

图只对固定 shape、整层 resident、`base_on_gpu=True` 生效；多 tile、换入层或
不同 down 对象自动回退到普通 stream 路径。图的 host 输入指针来自固定 pinned
buffer，每次 replay 只更新 buffer 内容，不重新创建图。

真实 Qwen3.8-27B Q4_K_M 单层 warm 测量，RTX 4070 Laptop：

```text
layer 3，含 down：普通路径约 1.93 ms，CUDA Graph 约 1.80 ms
layer 21，含 down：普通路径约 2.30 ms，CUDA Graph 约 2.20 ms
```

这是单层 gate/up/SwiGLU/down 链路，不是完整生成速度。图捕获主要减少重复
launch、事件和 stream 调度空隙；残差读取和 down 算术本身没有被“免费消除”。
首次 capture 仍需约数毫秒到十余毫秒，属于冷启动成本。

图路径同时通过了两个不同 activation 的重放校验，以及 fixture 和真实 Qwen
层的 down 数值校验。下一步要用 Nsight Systems/Compute 把 wall-time 改善拆成
H2D、kernel gap、SM active 和 down kernel 的实测证据，再决定是否作为默认策略。

本机 Nsight Compute 2022.3 可以启动，但驱动拒绝性能计数器访问
(`ERR_NVGPUCTRPERM`)，因此本轮没有伪造 occupancy 或 SM active 数字。脚本
`scripts/profile_resident_cuda_graph.py` 已加入，后续在开启 GPU performance
counter 权限的机器上可直接重放同样的采样。
