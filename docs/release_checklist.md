# GitHub Release Checklist

## 当前 release candidate

- [x] 数学报告：docs/math_principles_report.md
- [x] 原始公式规格：docs/weight_code_split_spec.md
- [x] Qwen3.8-27B 适配预算与未验证边界
- [x] CPU base + GPU residual + SwiGLU/down 桥接代码
- [x] 旧手机集群分析模拟
- [x] MIT License
- [x] 贡献指南
- [x] 迭代日志保留在 docs/log/
- [x] .gitignore 排除模型、运行库、二进制和生成结果

## 发布前本地检查

从仓库根目录运行：

~~~powershell
python -m compileall scripts tests
python -m pytest -q
git diff --check
~~~

CUDA 环境可额外运行：

~~~powershell
python scripts/build_and_run_full_ffn_cuda.py
python scripts/estimate_27b_ffn_budget.py --hidden 5120 --ffn 17408 --layers 64 --out results/qwen38_27b_budget.json
python scripts/simulate_phone_ffn_cluster.py
~~~

生成的 results/ 不应提交；它们用于本地复核和 release notes 摘要。

## GitHub 发布边界

发布说明必须明确：

- 这是 FFN 层级资源交换实验，不是完整模型加速器；
- Qwen3.8-27B 目前只有结构兼容和预算分析，尚未完成真实 checkpoint 的 8 GiB 端到端验证；
- exact residual 路径与近似 residual 路径必须分开报告；
- 手机集群结果是分析模拟，不能当作真实 Android benchmark；
- 需要用户或维护者提供 GitHub remote 后才能执行 push。

## 建议 tag

~~~text
v0.1.0-rc4
~~~

该 tag 表示“公开数学原理、实验脚本和已知边界”，不表示生产就绪。
