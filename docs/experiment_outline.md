# 前馈层实验大纲

日期：2026-09-02

## 1. 当前目标（2026-09-02 收敛版）

先解决单层/分层 FFN 的内存带宽和 GPU 计算不平衡，不追求全模型统一公式：

1. 每层 FFN 是否存在可利用的输入状态局部性。
2. 冷启动完整扫描一次原始 FFN 权重后，能否编译成运行时只读预展开 artifact、完全不再扫描 `Wg/Wu/Wd` 的展开算子。
3. 完整无损残差包能否触发足够多的 GPU 连续计算并恢复可接受精度。
4. 分层超级块搬运是否能降低 H2D 流量、显存占用和等待时间。
5. CPU 路由成本是否低于节省的 GPU/PCIe 成本。

最高原则：不追求 FLOPs 下降，允许总计算量增加。主机内存用于冷启动预展开，不作为首要优化目标；显存峰值不要求下降，只要不 OOM。主要优化方向改为 GPU 等待占比下降、copy/compute 重叠上升、持续算子利用率上升；每 token H2D 字节和显存峰值作为辅助账本。

暂不要求：全模型统一公式、跨层精确合并、KV cache 优化、MoE 分组和完整推理框架重写。

当前试验模型是 Qwen3.5 2B Q4_K_M。GGUF 元数据显示它是 `qwen35` 混合结构，24 层、hidden size 2048、FFN intermediate size 6144，并包含 SSM/线性注意力张量；因此首轮 FFN 实验要按层记录 Attention/SSM 类型，不能假设所有 block 相同。离线布局扫描显示该模型 FFN 张量约占 GGUF 文件 43.20%（523.13 MiB），不是纯 Transformer 中常见的参数占比，后续收益估算必须使用实测占比。layer 23 的精确 radix 表与 32-token holdout 已通过，下一步转入 C++/CUDA 流水线实测。

## 2. 分层工作假设

标准 FFN：

```text
F(x) = W2 * activation(W1 * x + b1) + b2
```

每层独立的预展开与残差校正：

```text
k_l = route_l(x_l)
F_hat_l(x_l) = B_l[k_l] + U_l[k_l] * alpha_l(x_l, c_l[k_l])
```

其中每层的 `c_l`、`B_l`、`U_l` 和门控阈值离线生成。第一版只接受轻量合并：`base + residual`；如果合并开销或误差不合格，立即回退整层精确 FFN。

### 2.1 主线改正：权重算术分解，不以状态码本替代输入

本项目的主线不是用激活中心或 VQ 码本近似 `x`。那类方法可作为误差基线，但不符合“先展开固定权重、只搬运锯齿残差”的目标。

对每层、每个输入/输出分区，都允许记录一条独立的拆分与合并规则：

```text
W_p = H_p + R_p
g = Merge_p(EvalBase_p(x, H_p), EvalResidual_p(x, R_p))
u = Merge_p(EvalBase_p(x, H'_p), EvalResidual_p(x, R'_p))
y = Wd * (SiLU(g) * u)
```

公式不要求是全局统一的线性式；可以使用定点位拆分、分段、取整/进位、位平面或每分区的公式表。验收的第一层是：在选定量化整数域内，`Merge` 必须重建与原投影相同的 `g/u`；只有明确声明的量化误差可以存在。

以进制拆分为例，令 `x = Bx*x_hi + x_lo`、`w = Bw*w_hi + w_lo`，点积必须保留四个交叉项：

```text
x*w = Bx*Bw*sum(x_hi*w_hi)
    + Bx*sum(x_hi*w_lo)
    + Bw*sum(x_lo*w_hi)
    +    sum(x_lo*w_lo)
```

加法例子中的进位在这里由 `Merge` 的尺度和累加规则承担，不能省略高位或交叉项。

关键系统约束：主机 RAM 中可以保留完整预展开产物；冷启动允许扫描原始 `Wg/Wu/Wd`。运行阶段只能执行已编译的有限公式、部分和或状态表，并读取 residual tile；**不能顺序读取任意稠密 `H_p` 的原码流或展开码流**。若 `H_p` 不能被编译为这种形式，它只能作为冷启动参考，不是候选运行时主项。后续分别测量状态表访问、聚合公式和残差 GPU 计算。`R_p` 才是需要高计算密度执行的锯齿残差。

预展开不等于查表：查表只是运行时取值方式之一。预展开到 RAM 后，可以直接按层/块地址读取基础项，或读取预排布的分段参数，再计算非线性差值；只有需要状态分类时才使用 LUT/中心查找。后续必须并行比较：

```text
direct block read + residual
indexed lookup + residual
piecewise formula + residual
```

非查表主线新增公式族：

```text
h_j = u_j * p_j((g_j-mu_j)/sigma_j)
y = sum_b W_down,b @ h_b
```

该公式把 SwiGLU 的 SiLU 非线性变成离线系数和运行时乘加；它是合并公式候选，不自动等价于权重搬运优化。必须与分块权重拆分联合评估。

权重不直接做跨非线性相加。优先把“权重预展开”理解为：权重分页、量化布局、分区算术基项、残差基底和公式表的离线展开；只有在 `g/u` 投影已等价或误差受控时，才能进入原始 SiLU 和 down 投影。

分块线性查表候选：

```text
x = [x1, x2, ..., xm]
W1*x ~= sum_j table[j, quantize(xj)]
```

非线性交叉项不丢弃，进入残差或精确回退。

## 3. 实验阶段

### 阶段 A：运行环境基线

- 确认 CPU、GPU、CUDA、PCIe、llama.cpp 版本。
- 对 Qwen3.5 2B Q4_K_M 跑 CPU、GPU、混合层数基线。
- 记录 prompt、decode、显存、吞吐、首 token 延迟和功耗（可得时）。

### 阶段 B：FFN 可压缩性扫描

- 获取可观测的 FFN 输入、门控输出和 FFN 输出。
- 按层统计相邻 token 差分、PCA 有效秩、聚类失真和残差能量。
- 分开统计 prefill 与 decode，先以 batch=1 decode 为主。

### 阶段 C：单层算子展开原型

- `K=1`：单中心查表。
- `K=16/64/256`：状态分区查表。
- 残差 rank 测试：`r=16/32/64/128`。
- 先只做 layer 23；layer 22 作为负对照。
- 共享基底、分层基底和每区域基底对比。
- 加入距离/残差阈值和整层精确 FFN 回退。
- 合并只比较 `base + residual` 与直接输出两种轻量路径。
- 查表不是必选组件；增加“直接取预展开块”的对照组，避免把分类和随机访问成本误算成算法必要成本。
- 增加“按中间神经元多项式 + down 分块合并”对照组；它不做运行时查表，也不调用 SiLU。
- 权重块选择使用真实激活敏感度，而不是只按 SVD/Frobenius 误差排序：`score_b = E_x ||delta_y,b(x)|| / ||y(x)||`。
- 新增精确算术分解基线：按层/分区构造 `H_p + R_p`，先证明 `Merge(EvalBase, EvalResidual) == W_p*x`，再记录 `H_p` 的结构化字节数、`R_p` 的运行时读取字节数和 GPU MAC/动态字节。
- 公式表的粒度为 `layer_id + projection + output_chunk + input_block`；表记录拆分格式、缩放/进位规则、执行位置和精确回退条件。它不是输入状态码本。
- 在进入 DMA、分页、H2D 或 kernel 基准前，先完成 `docs/weight_code_split_spec.md` 的数学验证：Q4_K 整数域精确重构、允许误差的残差界，以及每分区公式表的离线选择目标。

### 阶段 D：分层分页和块搬运

- 逻辑页：4KB/16KB/64KB，仅作为索引单位。
- 搬运超级块：256KB、1MB、4MB、16MB 对比。
- 使用 pinned host ring buffer、双缓冲和异步 H2D。
- 统计每层请求合并率、实际搬运字节、PCIe 往返次数和同步等待。
- 先测单层连续 token，再测相邻稳定层；不做跨层混合页。

### 阶段 E：分层端到端验证

- 单层误差：相对 L2、余弦相似度、最大分量误差。
- 模型误差：先测替换单层后的 next-token KL、logit 误差；稳定后再测多层组合。
- 系统指标：首 token、tokens/s、显存峰值、CPU 占用、RAM 占用。
- 每层可独立选择查表、低秩残差、LUT 或完整 FFN。

## 4. 当前验收门槛

以下是工程门槛，不是理论结论：

- 分层 `base + residual` 的误差明显低于纯查表。
- 在明确误差预算下，至少一个层达到可接受近似比例。
- 超级块搬运的有效带宽高于细粒度随机搬运。
- CPU 分类、合并和同步开销没有吃掉 FFN 节省。
- 非查表公式在目标层达到误差门槛，并能与分块权重拆分组合。
- 异常输入能稳定回退，不能让误差跨层失控。
- 冷启动原始 gate/up/down 权重完整扫描次数允许为一次；正常近似路径对原始权重及其等价展开主项流的完整扫描次数必须为零，只能访问有限公式、状态表和 residual tile。精确回退是否重新读取原始权重必须单独计数。
- 报告 `GPU resident peak`、`H2D bytes/token`、`GPU MAC or FLOP/H2D byte`、copy/sync 等待时间、copy/compute overlap 和 SM/算子利用率；主机预展开产物大小只作附注，不能代替这些指标。只要显存不 OOM，resident peak 不必低于原始路径。

## 5. 失败条件

- 输入状态没有局部性，任何合理 `K` 都需要接近完整表。
- 残差 rank 接近 `d_model`，无法减少计算。
- CPU/RAM 查表导致每层同步，端到端延迟反而增加。
- 校准分布外输入频繁触发精确回退。

## 6. 预展开产物

每个模型和硬件配置生成一个版本化目录：

```text
manifest.json
layout/
tables/
residual_bases/
thresholds/
fallback/
```

产物绑定模型哈希、量化方式、校准集、GPU 架构和块大小。

## 7. 后置问题

- KV cache 和长上下文显存管理。
- MoE 专家分组。
- 跨层共享公式和全局统一合并。
- llama.cpp 全图调度器重写。
