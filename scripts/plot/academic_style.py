"""统一学术绘图规范（候选全局 skill）。

设计目标：让 matplotlib 默认图摆脱"玩具感"，达到期刊/会议插图水准。
核心约定：
  - 矢量优先：同时导出 .pdf（论文嵌入）与 .png（预览）。
  - 去图内标题：标题交给 LaTeX \\caption；图内只留坐标轴/图例。
  - 去顶/右边框，仅留 L 型坐标轴；浅色横向网格在数据之下。
  - 方法固定配色，全篇一致；衬线字体 + Computer Modern mathtext。

用法：
    from academic_style import apply_style, method_color, add_bar_labels, save_figure
    apply_style()
    ...
    save_figure(fig, out_dir / "name")   # 自动产出 name.pdf 与 name.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- 方法固定配色（seaborn muted，低饱和，打印友好）----
METHOD_COLORS = {
    "full": "#4C72B0",    # 蓝
    "lora": "#DD8452",    # 橙
    "pissa": "#55A868",   # 绿
    "qlora": "#C44E52",   # 红
    "qpissa": "#8172B3",  # 紫
}
_FALLBACK_CYCLE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                   "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

# 方法显示名（图例/坐标轴用，大小写规范）
METHOD_LABELS = {
    "full": "Full FT",
    "lora": "LoRA",
    "pissa": "PiSSA",
    "qlora": "QLoRA",
    "qpissa": "QPiSSA",
}


def apply_style() -> None:
    """全局应用学术绘图 rcParams。所有绘图脚本开头调用一次即可。"""
    plt.rcParams.update({
        # 画布与导出
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # 字体（衬线 + CM mathtext，与 LaTeX 正文协调）
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "CMU Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # 坐标轴：去顶/右边框，留 L 型
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        # 网格：浅色横向，置于数据之下
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#B0B0B0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        # 线条与图例
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        # 字体可嵌入 PDF（避免编辑器显示异常）
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def method_color(name: str) -> str:
    key = name.lower()
    if key in METHOD_COLORS:
        return METHOD_COLORS[key]
    idx = abs(hash(key)) % len(_FALLBACK_CYCLE)
    return _FALLBACK_CYCLE[idx]


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name.lower(), name.upper())


def grid_y_only(ax) -> None:
    """只保留横向网格线（柱状/曲线读数更清爽）。"""
    ax.grid(axis="y", which="major")
    ax.grid(axis="x", visible=False)


def add_bar_labels(ax, bars, values, fmt="{:.1f}", dy_frac=0.012) -> None:
    """在柱顶添加数值标签，偏移按 y 轴量程比例自适应。"""
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) * dy_frac
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=10)


def save_figure(fig, path_stem, formats=("pdf", "png")) -> list[Path]:
    """同时导出多种格式（默认矢量 pdf + 预览 png）。path_stem 不带扩展名。"""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out
