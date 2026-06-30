"""[M1] 汇总作图：训练 loss 曲线对比 + GSM8K 准确率柱状图。

读取 results/m1/exp1/metrics/ 下已有的 train_*.json 与 eval_*.json，缺哪个跳过哪个。

用法：
    python scripts/m1/plot_curves.py --config configs/m1_llama2_7b.yaml
"""

from __future__ import annotations

import argparse
import json

from common import abspath, load_config

# 已知方法的固定绘图顺序；未列出的方法按字母序追加在后面。
METHOD_ORDER = ["full", "lora", "pissa", "qlora", "qpissa"]
_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _ordered_methods(metrics_dir, kind: str) -> list[str]:
    """根据 metrics_dir 下实际存在的 {kind}_*.json 自动检测方法并排序。"""
    found = []
    for p in metrics_dir.glob(f"{kind}_*.json"):
        found.append(p.stem[len(kind) + 1 :])
    known = [m for m in METHOD_ORDER if m in found]
    extra = sorted(m for m in found if m not in METHOD_ORDER)
    return known + extra


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总作图：训练 loss 曲线 + GSM8K 准确率柱状图")
    parser.add_argument("--config", default="configs/m1_llama2_7b.yaml")
    parser.add_argument("--prefix", default=None, help="输出图文件名前缀（默认取 experiment 名）")
    parser.add_argument("--title", default=None, help="图标题（默认取 experiment 名）")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(abspath(args.config))
    metrics_dir = abspath(cfg["paths"]["output_dir"]) / "metrics"
    fig_dir = abspath(cfg["paths"]["output_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    exp = cfg.get("experiment", "exp")
    prefix = args.prefix or exp
    title = args.title or exp

    # ---- loss 曲线 ----
    plt.figure(figsize=(7.5, 4.5))
    any_curve = False
    for method in _ordered_methods(metrics_dir, "train"):
        p = metrics_dir / f"train_{method}.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        curve = data["results"]["loss_curve"]
        if not curve:
            continue
        steps = [c["step"] for c in curve]
        losses = [c["loss"] for c in curve]
        plt.plot(steps, losses, label=method.upper())
        any_curve = True
    if any_curve:
        plt.xlabel("optimizer step")
        plt.ylabel("training loss")
        plt.title(f"{title} — training loss")
        plt.legend()
        plt.tight_layout()
        loss_path = fig_dir / f"{prefix}_loss_curves.png"
        plt.savefig(loss_path, dpi=150)
        print(f"已保存：{loss_path}")
    plt.close()

    # ---- 准确率柱状图 ----
    accs = {}
    for method in _ordered_methods(metrics_dir, "eval"):
        p = metrics_dir / f"eval_{method}.json"
        accs[method] = json.loads(p.read_text(encoding="utf-8"))["results"]["accuracy"]
    if accs:
        plt.figure(figsize=(5.5, 4.5))
        names = list(accs.keys())
        vals = [accs[n] * 100 for n in names]
        bars = plt.bar([n.upper() for n in names], vals, color=_COLORS[: len(names)])
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center")
        plt.ylabel("GSM8K accuracy (%)")
        plt.title(f"{title} — GSM8K accuracy by method")
        plt.tight_layout()
        acc_path = fig_dir / f"{prefix}_gsm8k_accuracy.png"
        plt.savefig(acc_path, dpi=150)
        print(f"已保存：{acc_path}")
        print("准确率：", {k: f"{v * 100:.2f}%" for k, v in accs.items()})
    plt.close()


if __name__ == "__main__":
    main()
