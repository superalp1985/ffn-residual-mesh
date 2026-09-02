# 调研与设计记录

日期：2026-09-02

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
