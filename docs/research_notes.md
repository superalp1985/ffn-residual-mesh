# 调研与设计记录

日期：2026-09-02

## 2026-09-03 最高优化原则

本项目不以减少 FLOPs 为目标。允许通过解码、张量收缩、交叉项和残差重构增加总计算量，换取：

1. 运行时动态搬运字节下降。
2. 每个搬运字节触发的 GPU 运算增加。
3. GPU 等待 PCIe、显存数据和同步的时间下降。

预展开必须是冷启动算子编译，而不是把完整 FFN 矩阵乘移到 CPU：

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
