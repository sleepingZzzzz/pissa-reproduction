"""[M3] Table 4：量化重建误差对比（QPiSSA vs QLoRA）。

用法：
    python scripts/m3/table4.py --config configs/m3_llama2_7b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import bitsandbytes as bnb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pissa import PiSSALinear, inject_pissa, merge_pissa
from pissa.svd_init import pissa_init, reconstruction_error_against

# 复用 M1 common 的配置/模型加载
sys.path.insert(0, str(REPO_ROOT / "scripts" / "m1"))
from common import (
    abspath,
    load_config,
    resolve_model_path,
    torch_dtype,
)

# PiSSA 初始化的 loRA_A/loRA_B 命名
LORA_A = "lora_A"
LORA_B = "lora_B"


def nf4_quant_dequant(w: torch.Tensor, compute_dtype=torch.float32) -> torch.Tensor:
    """NF4 量化→反量化。返回反量化浮点张量。"""
    w_fp = w.to(dtype=compute_dtype, copy=True).detach()
    # bitsandbytes 的 NF4 量化接口
    w_nf4, quant_state = bnb.functional.quantize_nf4(w_fp)
    w_dq = bnb.functional.dequantize_nf4(w_nf4, quant_state)
    return w_dq.to(dtype=w.dtype)


def linear_layers(model: nn.Module, target_modules: tuple[str, ...]) -> list[tuple[str, nn.Linear]]:
    return [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in target_modules
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="[M3] Table 4 量化误差")
    parser.add_argument("--config", default="configs/m3_llama2_7b.yaml")
    parser.add_argument("--layers", type=int, default=None, help="只测前 N 层（冒烟）")
    args = parser.parse_args()

    cfg = load_config(abspath(args.config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path(cfg)
    dtype = torch.float32  # 误差计算用 fp32

    # 加载模型（不含量化）
    from transformers import AutoModelForCausalLM

    print(f"加载模型 {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    target_modules = tuple(cfg["adapter"]["target_modules"])
    r = cfg["adapter"]["r"]
    qcfg = cfg["quantization"]
    compute_dtype = torch.float32

    lines = linear_layers(model, target_modules)
    if args.layers:
        lines = lines[: args.layers]
    print(f"待测线性层：{len(lines)}（target_modules={target_modules}）")

    results = []
    for name, module in lines:
        W = module.weight.data.detach().to(dtype=compute_dtype)
        try:
            # === 1. 基线量化误差（NF4 量化原权重）===
            W_q = nf4_quant_dequant(W, compute_dtype)
            err_baseline = (W - W_q).norm().item()

            # === 2. PiSSA SVD 分解 ===
            init = pissa_init(W, r=r, niter=None)
            err_svd = reconstruction_error_against(init, W)
            W_res = init.residual.to(dtype=compute_dtype)
            A = init.lora_A.to(dtype=compute_dtype)  # (r, in_features)
            B = init.lora_B.to(dtype=compute_dtype)  # (out_features, r)

            # === 3. 量化残差 ===
            W_res_q = nf4_quant_dequant(W_res, compute_dtype)
            W_pissa_rec = W_res_q + B @ A
            err_pissa = (W - W_pissa_rec).norm().item()

            # === 4. LoRA 初始化误差（A=rand, B=0 → 增量为 0）===
            # LoRA 初始增量为 0，所以重建 = NF4(W)
            err_lora = err_baseline  # 同上

            results.append({
                "layer": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "baseline_nf4_err": round(err_baseline, 6),
                "pissa_nf4_err": round(err_pissa, 6),
                "svd_recon_err": round(err_svd, 8),
            })

            rel_improve = (err_baseline - err_pissa) / max(err_baseline, 1e-12) * 100
            print(f"  {name}: baseline={err_baseline:.4f}  pissa={err_pissa:.4f}  "
                  f"(↓{rel_improve:.1f}%)  svd_err={err_svd:.2e}")

        except Exception as e:
            print(f"  {name}: SKIP ({e})")
            results.append({"layer": name, "error": str(e)})

    # ---- 汇总 ----
    valid = [r for r in results if "pissa_nf4_err" in r]
    if valid:
        avg_baseline = sum(r["baseline_nf4_err"] for r in valid) / len(valid)
        avg_pissa = sum(r["pissa_nf4_err"] for r in valid) / len(valid)
        summary = {
            "num_layers": len(valid),
            "avg_baseline_nf4_err": round(avg_baseline, 4),
            "avg_pissa_nf4_err": round(avg_pissa, 4),
            "avg_improvement_pct": round((avg_baseline - avg_pissa) / max(avg_baseline, 1e-12) * 100, 2),
        }
        print(f"\n总计：{len(valid)} 层, 平均 NF4 误差 {avg_baseline:.4f} → PiSSA {avg_pissa:.4f}"
              f"  (改善 {summary['avg_improvement_pct']:.1f}%)")
    else:
        summary = {"error": "no valid results"}

    # ---- 保存 ----
    out_dir = abspath(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table4_quant_error.json"
    payload = {
        "experiment": "M3/Table4 量化重建误差",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "r": r,
            "target_modules": list(target_modules),
            "quant_type": qcfg["bnb_4bit_quant_type"],
            "compute_dtype": str(compute_dtype),
        },
        "summary": summary,
        "per_layer": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已保存 {out_path}")


if __name__ == "__main__":
    main()
