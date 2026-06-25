# PiSSA 复现

复现论文 **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models**（NeurIPS 2024）。

PiSSA 与 LoRA 架构相同，区别仅在适配器初始化：对预训练权重 `W` 做 SVD，用前 `r` 个主成分初始化可训练的 `A`、`B`，其余作为冻结残差 `W_res`，从而比 LoRA 收敛更快、效果更好。

```
W = U S Vᵀ
A = U[:,:r] · S[:r]^½ ,  B = S[:r]^½ · V[:,:r]ᵀ ,  W_res = W − AB
forward:  y = x(W_res + AB)
```

## 复现里程碑

| 里程碑 | 内容 | 算力 |
|---|---|---|
| **M0** | 从零实现 SVD 初始化，验证正确性（重构恒等、与 PEFT 对齐、Fig.3 奇异值谱、Fig.2a MNIST 玩具收敛） | 本地 / 免费 |
| **M1** | 单模型 **Qwen3.5-2B** 上 PiSSA vs LoRA(±Full)，证明收敛更快更好（MetaMathQA→GSM8K） | 单卡 4090 |
| **M2** | 多模型 **Qwen3.5-9B / Llama-3.1-8B / Gemma-2-9B** 重复实验，证明泛用性（Table 2 思路） | 单卡 A100 等 |
| **M3** | 量化 QPiSSA：误差降低表（Table 4，纯计算）+ 4-bit 训练收敛（Fig.5） | 单卡 |

## 目录结构

```
src/pissa/          从零实现：svd_init / layer(PiSSALinear) / inject
scripts/m0..m3/     各里程碑可执行脚本
configs/            实验配置（base.yaml 为公共超参）
results/m*/exp*/    实验产物（图、指标）
.log/               项目记录（记录.md / 反思.md）+ report/（NeurIPS LaTeX 模板）
data/ checkpoints/  大文件，仅服务器，.gitignore
```

## 环境

```bash
pip install -r requirements.txt
```

## 运行 M0

```bash
# 1. 验证初始化正确性（重构恒等 + 与 PEFT 对齐）
python scripts/m0/verify_init.py --in-features 1024 --out-features 1024 --r 64

# 2. 奇异值谱（Fig.3）；无网络可加 --random
python scripts/m0/plot_singular.py --random

# 3. MNIST 玩具收敛对比（Fig.2a）
python scripts/m0/mnist_toy.py --epochs 3 --ft-steps 200
```

产物保存在 `results/m0/exp1/figures/`。

## 模型选型

- M1：`Qwen3.5-2B`（Base，dense，2026-03）
- M2：`Qwen3.5-9B`（2026-03） / `Llama-3.1-8B`（2024-07） / `Gemma-2-9B`（2024-06），dense + base，跨三家族证明泛用性。
