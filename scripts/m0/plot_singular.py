"""[M0] 复现论文 Fig.3 风格：可视化某层权重 W 与残差 W_res 的奇异值谱。

可从 ModelScope / HuggingFace 加载某模型某一层的真实权重；为避免下载整模型（8GB），
只用 HTTP Range 读取目标张量所在的字节（约几十 MB）。若 ``--random`` 或联网失败，
回退到随机矩阵，仍可演示"主成分被移除后残差奇异值分布变窄"的现象。

Qwen3.5 为多模态 + 混合注意力（3:1 Gated DeltaNet / Gated Attention），带标准
注意力的层在索引 3/7/11/15/19/23，权重键前缀为 model.language_model.layers。

用法：
    python scripts/m0/plot_singular.py --model Qwen/Qwen3.5-2B-Base \
        --layer model.language_model.layers.11.self_attn.q_proj --r 128 --source auto
    python scripts/m0/plot_singular.py --random  # 无需联网
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pissa import pissa_init  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "m0" / "exp1" / "figures"
METRICS_DIR = REPO_ROOT / "results" / "m0" / "exp1" / "metrics"

# safetensors dtype 字符串 -> torch dtype
_ST_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
}


def _base_url(source: str, repo: str) -> str:
    if source == "modelscope":
        return f"https://modelscope.cn/models/{repo}/resolve/master"
    if source == "hf":
        return f"https://huggingface.co/{repo}/resolve/main"
    raise ValueError(source)


def _range_get(url: str, start: int, end: int, timeout: int = 30) -> bytes:
    """HTTP Range 读取 [start, end]（含端点）字节。"""
    import requests

    headers = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    return r.content


def _load_tensor_via_range(base: str, filename: str, key: str) -> torch.Tensor:
    """仅用 HTTP Range 从远端 safetensors 文件中读取单个张量。

    safetensors 布局：[8B 小端 uint64 头长 N][N 字节 JSON 头][数据区]。
    JSON 头记录每个张量的 dtype/shape/data_offsets(相对数据区)。
    """
    import requests  # noqa: F401  确保依赖存在

    url = f"{base}/{filename}"
    header_len = struct.unpack("<Q", _range_get(url, 0, 7))[0]
    header = json.loads(_range_get(url, 8, 8 + header_len - 1).decode("utf-8"))
    if key not in header:
        raise KeyError(f"张量键 {key} 不在 safetensors 头中")
    meta = header[key]
    dtype = _ST_DTYPE[meta["dtype"]]
    begin, end = meta["data_offsets"]
    data_start = 8 + header_len
    raw = _range_get(url, data_start + begin, data_start + end - 1)
    t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(meta["shape"])
    return t.to(torch.float32)


def _resolve_key(base: str, layer: str) -> tuple[str, str]:
    """读 index.json，返回 (张量键, 所在文件名)。layer 可不带 .weight。"""
    import requests

    key = layer if layer.endswith(".weight") else layer + ".weight"
    r = requests.get(f"{base}/model.safetensors.index.json", timeout=30)
    if r.status_code == 200:
        weight_map = r.json()["weight_map"]
        if key in weight_map:
            return key, weight_map[key]
        cands = sorted(k for k in weight_map if k.endswith("self_attn.q_proj.weight"))
        if cands:
            print(f"  未找到 {key}，改用 {cands[0]}")
            return cands[0], weight_map[cands[0]]
        raise KeyError(f"index.json 中找不到 {key}")
    # 无 index（单文件无分片）时，默认单文件名。
    return key, "model.safetensors"


def load_weight(args: argparse.Namespace) -> torch.Tensor:
    if args.random:
        torch.manual_seed(0)
        return torch.randn(args.out_features, args.in_features)

    sources = (
        ["modelscope", "hf"] if args.source == "auto" else [args.source]
    )
    last_err: Exception | None = None
    for src in sources:
        try:
            base = _base_url(src, args.model)
            print(f"[{src}] 解析 {args.model} 的层 {args.layer} ...")
            key, filename = _resolve_key(base, args.layer)
            print(f"[{src}] 通过 HTTP Range 读取张量 {key}（文件 {filename}）...")
            w = _load_tensor_via_range(base, filename, key)
            print(f"[{src}] 读取成功，shape={tuple(w.shape)}")
            return w
        except Exception as exc:  # noqa: BLE001
            print(f"[{src}] 失败：{type(exc).__name__}: {exc}")
            last_err = exc
    print(f"所有源均失败（{last_err}），回退随机矩阵。")
    torch.manual_seed(0)
    return torch.randn(args.out_features, args.in_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="[M0] 奇异值谱可视化（Fig.3）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument(
        "--layer",
        type=str,
        default="model.language_model.layers.11.self_attn.q_proj",
        help="层路径（可不带 .weight）；Qwen3.5 注意力层在索引 3/7/11/15/19/23",
    )
    parser.add_argument("--r", type=int, default=128)
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "modelscope", "hf"],
        help="下载源；auto 先试 ModelScope 再试 HuggingFace",
    )
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

    # 主成分保留的能量占比：sum(s[:r]^2) / sum(s^2)。
    energy_total = float((s_full**2).sum())
    energy_principal = float((s_full[: args.r] ** 2).sum())
    energy_ratio = energy_principal / max(energy_total, 1e-12)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "singular.json"
    payload = {
        "experiment": "M0/plot_singular (Fig.3)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "model": "random" if args.random else args.model,
            "layer": None if args.random else args.layer,
            "source": None if args.random else args.source,
            "random": bool(args.random),
            "r": args.r,
            "weight_shape": list(weight.shape),
            "figure": str(out_path),
        },
        "results": {
            "W_singular_max": float(s_full.max()),
            "W_singular_min": float(s_full.min()),
            "W_res_singular_max": float(s_res.max()),
            "principal_energy_ratio": energy_ratio,
            "W_singular_values": s_full.tolist(),
            "W_res_singular_values": s_res.tolist(),
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存 metrics：{metrics_path}")


if __name__ == "__main__":
    main()
