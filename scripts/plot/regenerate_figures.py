"""按统一学术规范（academic_style）批量重绘论文图表。

数据源全部为 results/**/metrics/*.json（已跑完的实验产物），重新出图：
  - M1：训练 loss 曲线 + GSM8K 准确率柱状
  - M3：训练 loss 曲线 + GSM8K 准确率柱状（QPiSSA vs QLoRA）
  - M2：跨模型 PiSSA vs LoRA 分组柱状（含 20K 受限规模标注）
  - M0：奇异值谱（双面板）+ MNIST toy 收敛曲线

每图同时导出 .pdf（论文嵌入）与 .png（预览）。

用法：
    python scripts/plot/regenerate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from academic_style import (
    apply_style,
    method_color,
    method_label,
    add_bar_labels,
    save_figure,
    grid_y_only,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

METHOD_ORDER = ["full", "lora", "pissa", "qlora", "qpissa"]


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _present_methods(metrics_dir: Path, kind: str) -> list[str]:
    found = [p.stem[len(kind) + 1:] for p in metrics_dir.glob(f"{kind}_*.json")]
    known = [m for m in METHOD_ORDER if m in found]
    return known + sorted(m for m in found if m not in METHOD_ORDER)


def _ema(values: list[float], alpha: float = 0.15) -> list[float]:
    """指数滑动平均（带偏差校正），用于平滑训练 loss。alpha 越小越平滑。"""
    out, m = [], 0.0
    for i, v in enumerate(values):
        m = v if i == 0 else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def plot_loss_curves(metrics_dir: Path, out_stem: Path, alpha: float = 0.15) -> None:
    methods = _present_methods(metrics_dir, "train")
    if not methods:
        return
    fig, ax = plt.subplots()
    drew = False
    for m in methods:
        curve = _read(metrics_dir / f"train_{m}.json")["results"]["loss_curve"]
        if not curve:
            continue
        steps = [c["step"] for c in curve]
        losses = [c["loss"] for c in curve]
        color = method_color(m)
        # 原始曲线：浅色细线作背景，保留真实波动
        ax.plot(steps, losses, color=color, lw=0.8, alpha=0.22)
        # 平滑曲线：同色实线，主视觉
        ax.plot(steps, _ema(losses, alpha), label=method_label(m), color=color)
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss")
    ax.legend()
    grid_y_only(ax)
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f"[loss] {out_stem}.pdf/.png")


def plot_accuracy_bar(metrics_dir: Path, out_stem: Path) -> None:
    methods = _present_methods(metrics_dir, "eval")
    accs = []
    for m in methods:
        acc = _read(metrics_dir / f"eval_{m}.json")["results"]["accuracy"] * 100
        accs.append((m, acc))
    accs = [(m, a) for m, a in accs if a > 0]  # 跳过失败(0)的方法
    if not accs:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    names = [method_label(m) for m, _ in accs]
    vals = [a for _, a in accs]
    colors = [method_color(m) for m, _ in accs]
    bars = ax.bar(names, vals, color=colors, width=0.62,
                  edgecolor="white", linewidth=0.8)
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_ylim(0, max(vals) * 1.18)
    add_bar_labels(ax, bars, vals)
    grid_y_only(ax)
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f"[acc ] {out_stem}.pdf/.png")


def plot_m2_grouped(out_stem: Path) -> None:
    """M2 跨模型 PiSSA vs LoRA 分组柱状。20K 受限模型在标签下标注 †。"""
    models = [
        ("m2_chatglm3_6b", "ChatGLM3-6B", False),
        ("m2_qwen25_7b", "Qwen2.5-7B", True),
        ("m2_llama3_8b", "Llama3-8B", True),
    ]
    import numpy as np

    labels, pissa_v, lora_v = [], [], []
    for d, disp, partial in models:
        md = RESULTS / d / "exp1" / "metrics"
        try:
            pv = _read(md / "eval_pissa.json")["results"]["accuracy"] * 100
            lv = _read(md / "eval_lora.json")["results"]["accuracy"] * 100
        except FileNotFoundError:
            continue
        labels.append(disp + ("\n(20K †)" if partial else "\n(100K)"))
        pissa_v.append(pv)
        lora_v.append(lv)
    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    b1 = ax.bar(x - w / 2, pissa_v, w, label="PiSSA",
                color=method_color("pissa"), edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, lora_v, w, label="LoRA",
                color=method_color("lora"), edgecolor="white", linewidth=0.8)
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(pissa_v + lora_v) * 1.18)
    add_bar_labels(ax, b1, pissa_v)
    add_bar_labels(ax, b2, lora_v)
    ax.legend()
    grid_y_only(ax)
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f"[m2  ] {out_stem}.pdf/.png")


def plot_m0_singular(out_stem: Path) -> None:
    p = RESULTS / "m0" / "exp1" / "metrics" / "singular.json"
    if not p.exists():
        return
    data = _read(p)["results"]
    s_full = data["W_singular_values"]
    s_res = data["W_res_singular_values"]
    r = _read(p)["config"].get("r", 128)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(s_full, color=method_color("full"), label=r"$W$")
    axes[0].axvline(r, color="#C44E52", ls="--", lw=1.2, label=f"$r={r}$")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Singular value")
    axes[0].legend()
    axes[1].plot(s_res, color=method_color("pissa"), label=r"$W^{res}$")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Singular value")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f"[m0-s] {out_stem}.pdf/.png")


def plot_m0_mnist(out_stem: Path) -> None:
    p = RESULTS / "m0" / "exp1" / "metrics" / "mnist.json"
    if not p.exists():
        return
    curves = _read(p)["results"]["loss_curves"]
    fig, ax = plt.subplots()
    for m in ["full", "lora", "pissa"]:
        if m in curves:
            ax.plot(curves[m], label=method_label(m), color=method_color(m))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss")
    ax.legend()
    grid_y_only(ax)
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f"[m0-m] {out_stem}.pdf/.png")


def main() -> None:
    apply_style()

    m1 = RESULTS / "m1" / "exp1"
    plot_loss_curves(m1 / "metrics", m1 / "figures" / "m1_loss_curves")
    plot_accuracy_bar(m1 / "metrics", m1 / "figures" / "m1_gsm8k_accuracy")

    m3 = RESULTS / "m3_qpissa" / "exp1"
    plot_loss_curves(m3 / "metrics", m3 / "figures" / "m3_qpissa_loss_curves")
    plot_accuracy_bar(m3 / "metrics", m3 / "figures" / "m3_qpissa_gsm8k_accuracy")

    plot_m2_grouped(RESULTS / "m2_chatglm3_6b" / "exp1" / "figures" / "m2_grouped_accuracy")

    # M2 单模型图（Qwen2.5 / Llama3，含 PiSSA vs LoRA）
    for d in ["m2_qwen25_7b", "m2_llama3_8b"]:
        base = RESULTS / d / "exp1"
        plot_loss_curves(base / "metrics", base / "figures" / "m1_loss_curves")
        plot_accuracy_bar(base / "metrics", base / "figures" / "m1_gsm8k_accuracy")

    m0 = RESULTS / "m0" / "exp1"
    plot_m0_singular(m0 / "figures" / "singular_values")
    plot_m0_mnist(m0 / "figures" / "mnist_toy_convergence")

    print("完成。所有图已重绘为 .pdf + .png。")


if __name__ == "__main__":
    main()
