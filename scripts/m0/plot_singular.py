"""[M0] 复现论文 Fig.3 风格：可视化某层权重 W 与残差 W_res 的奇异值谱。

默认从 HuggingFace 加载一个模型的某个线性层；若未安装 transformers 或无法联网，
回退到随机矩阵，仍可演示"主成分被移除后残差奇异值分布变窄"的现象。

用法：
    python scripts/m0/plot_singular.py --model Qwen/Qwen3.5-2B --layer model.layers.0.self_attn.q_proj --r 128
    python scripts/m0/plot_singular.py --random  # 无需联网
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pissa import pissa_init  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "m0" / "exp1" / "figures"


def load_weight(args: argparse.Namespace) -> torch.Tensor:
    if args.random:
        torch.manual_seed(0)
        return torch.randn(args.out_features, args.in_features)
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        print(f"未安装 transformers（{exc}），回退随机矩阵。")
        torch.manual_seed(0)
        return torch.randn(args.out_features, args.in_features)

    print(f"加载 {args.model} 的 {args.layer} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    weight = model.get_submodule(args.layer).weight.data.clone()
    return weight


def main() -> None:
    parser = argparse.ArgumentParser(description="[M0] 奇异值谱可视化（Fig.3）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--layer", type=str, default="model.layers.0.self_attn.q_proj")
    parser.add_argument("--r", type=int, default=128)
    parser.add_argument("--random", action="store_true", help="用随机矩阵，免联网")
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=2048)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weight = load_weight(args)
    init = pissa_init(weight, r=args.r)
    s_full = init.singular_values.float().cpu().numpy()

    # 残差的奇异值（前 r 个主成分已被移除）。
    s_res = torch.linalg.svdvals(init.residual.float()).cpu().numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(s_full, label="W (full)")
    axes[0].axvline(args.r, color="r", ls="--", label=f"r={args.r}")
    axes[0].set_title("Singular values of W")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("magnitude")
    axes[0].legend()

    axes[1].plot(s_res, color="tab:orange", label="W_res")
    axes[1].set_title("Singular values of W_res (principal removed)")
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("magnitude")
    axes[1].legend()

    fig.tight_layout()
    out_path = OUT_DIR / "singular_values.png"
    fig.savefig(out_path, dpi=150)
    print(f"已保存：{out_path}")
    print(f"W   最大/最小奇异值: {s_full.max():.4f} / {s_full.min():.4e}")
    print(f"W_res 最大奇异值:    {s_res.max():.4f}")


if __name__ == "__main__":
    main()
