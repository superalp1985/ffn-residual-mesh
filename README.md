# FFN Residual Lab

本项目研究一种面向本地大模型推理的 FFN 优化路线：

> 离线预展开算子，使用 CPU/RAM 保存较大的基础表示，GPU 只计算小残差，按块异步搬运并合并。

首个实验资产来自 `E:\AccountingDemo-小企业会计`：Qwen3.5 2B Q4_K_M GGUF 与两套 llama.cpp 运行库。模型和运行库保留在本机，但被 `.gitignore` 排除，不进入 Git 提交。

## 研究边界

- 第一阶段只研究 Transformer 前馈层（FFN/SwiGLU）。
- Attention、KV cache 和完整模型重写暂不纳入首轮目标。
- 不以 MoE 专家分组作为前提，先研究状态查表、算子展开、残差近似和块搬运。
- 所有近似路径必须保留精确 FFN 回退路径。

## 目录

- `docs/experiment_outline.md`：实验总纲和验收门槛。
- `docs/research_notes.md`：调研结论、公式和工程约束。
- `docs/asset_manifest.md`：模型、运行库和环境来源。
- `docs/log/`：按日期记录每轮实验。
- `scripts/`：可重复运行的探针和基线脚本。
- `src/`：实验代码。

## Git 约定

每轮实验完成后记录：假设、命令、硬件、输入、结果、结论和下一步。大文件只记录来源、大小、哈希和版本，不提交二进制资产。
