# FFN Residual Lab

本项目研究一种面向本地大模型推理的 FFN 优化路线：

> 冷启动可把 FFN 预展开到主机内存；运行时只向 GPU 传紧凑差值包，让 GPU 用更多连续计算换取更低的显存占用和 PCIe 带宽需求。

首个实验资产来自 `E:\AccountingDemo-小企业会计`：Qwen3.5 2B Q4_K_M GGUF 与两套 llama.cpp 运行库。模型和运行库保留在本机，但被 `.gitignore` 排除，不进入 Git 提交。

## 研究边界

- 第一阶段只研究 Transformer 前馈层（FFN/SwiGLU）。
- Attention、KV cache 和完整模型重写暂不纳入首轮目标。
- 不以 MoE 专家分组作为前提，先研究状态查表、算子展开、残差近似和块搬运。
- 所有近似路径必须保留精确 FFN 回退路径。
- 当前只做分层 FFN：每层独立预展开、路由、残差计算和轻量合并。
- 暂不研究 KV cache、MoE 分组、跨层精确合并或全局统一公式。
- 预展开与查表分开评估：预展开是内存布局和离线产物，查表只是可选的运行时选择机制。
- 冷启动编译阶段允许完整扫描原始 gate/up/down 权重，并把拆分后的基项、残差、尺度、公式和索引写入主机内存 artifact；正常运行路径不得再次扫描原始权重，**也不得顺序扫描等价的已展开主项权重流**。主项必须被编译为有限公式、电路或按输入块状态索引的部分和；原始权重运行时只保留给明确触发的精确回退路径。
- 残差默认保持数学上完整，只允许无损位打包；裁剪、舍弃或再次量化残差属于独立的有损实验，不得混入精确拆分结论。
- 主机内存占用不是首要约束，允许冷启动展开和长期驻留；显存峰值也不要求下降，只要不超过硬件容量并能提供足够的双缓冲/预取空间。
- FLOPs 减少不是目标。最高指标是 GPU 等待占比下降、copy/compute 重叠上升、持续算子利用率上升；H2D 字节和显存峰值是约束与辅助指标，不是唯一目标。

## 当前计划

数学拆分和 layer 23 的 CUDA 流水线已完成。当前运行时 artifact 由两部分组成：CPU 读取有限 radix partial-sum table，GPU 读取无损 2-bit residual tile；不再打开 GGUF 或顺序扫描展开主项。1024-row 双缓冲残差路径已通过 CUDA 实测，CPU 主项 + GPU 残差并行在 8--12 个 CPU 线程时取得约 1.5x 的 layer-level critical-path 改善。

完整桥接 runner 已接入 `base + residual -> SwiGLU -> down`：layer 23 使用 9 MiB gate/up residual 包和 49 KiB CPU base 辅助结果，在 RTX 4070 上 2048-row 双缓冲 critical path 约比串行低 1.39x，完整输出相对 fp32 参考约 `1.18e-4`。这仍是层级资源交换证据，不宣称端到端模型加速。Python table mode 已证明只适合作为正确性 oracle；目标运行时的 CPU base 生产继续使用 C++ 常驻线程池。

当前资源放置原则已固定：主机内存可以多留预展开数据，显存可以保留较大的残差窗口、预取页和双/三缓冲工作集，只要不造成 OOM。任何方案必须报告显存峰值、权重 H2D 摊销、base 输出交换、GPU 等待、copy/compute 重叠和持续算子利用率，不能只报告静态压缩率。

## 目录

- `docs/experiment_outline.md`：实验总纲和验收门槛。
- `docs/research_notes.md`：调研结论、公式和工程约束。
- `docs/math_principles_report.md`：公开数学原理、运行时边界和 Qwen3.8-27B 适配报告。
- `docs/comfyui_phone_cluster_design.md`：MiniMax H3/ComfyUI 手机或平板 FFN worker 后端设计与模拟结论。
- `docs/release_checklist.md`：GitHub release candidate 检查项与发布边界。
- `docs/asset_manifest.md`：模型、运行库和环境来源。
- `docs/log/`：按日期记录每轮实验。
- `scripts/`：可重复运行的探针和基线脚本。
- `src/`：实验代码。
- `src/phone_ffn_loopback.py`：手机/平板 FFN worker 的协议级 loopback，覆盖分片、校验、并发和 deadline fallback。

本轮新增：`scripts/estimate_27b_ffn_budget.py` 用于 27B-class FFN 的显存、H2D 和计算密度账本；`scripts/build_and_run_full_ffn_cuda.py` 与 `src/exact_cpu_base_gpu_full_ffn_runner.cu` 用于 C++ CPU 主项 + GPU residual/SwiGLU/down 双缓冲验证。预算结果只代表单层/FFN 子图，不代表完整 27B 端到端吞吐。

另有 `scripts/simulate_phone_ffn_cluster.py`，用于模拟旧手机组成分布式 FFN 主项计算池。当前默认模型为输出行分片：手机保存编译表分片并并行生成 gate/up base，中心 GPU 保持 residual、合并、SwiGLU 和 down。该模拟只用于网络、L3 工作集和临界路径预算，不代表真实 Android 性能。

MiniMax H3/ComfyUI 的视频场景使用 `scripts/simulate_comfyui_phone_ffn.py`。该脚本显式区分精确 gate/up 回传与 hidden 近似回传，并把 TeaCache 的 real/reuse step 计入网络和 residual 账本。

`src/phone_ffn_loopback.py` 目前只验证调度协议，不代表真实 Android 性能；它把 worker 超时和校验失败都导向中心 GPU 的精确 fallback。

## Qwen3.8-27B 状态

Qwen3.8-27B 的 FFN 结构与本项目兼容，但当前仓库尚未编译该模型的真实 artifact，也没有完成 8 GiB 显卡上的端到端实测。按官方配置 `hidden_size=5120`、`intermediate_size=17408`、`num_hidden_layers=64` 的预算结果见 `docs/math_principles_report.md`；运行时仍需逐层驻留/流式分页，不能把全部 down 权重同时放入 8 GiB 显存。

本仓库目前适合作为公开实验协议、数学基线和复现实验脚本。它不宣称已经完成 27B 模型加速或生产部署。

## Git 约定

每轮实验完成后记录：假设、命令、硬件、输入、结果、结论和下一步。大文件只记录来源、大小、哈希和版本，不提交二进制资产。
