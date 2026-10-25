"""[M3] Table 4：量化重建误差对比（QPiSSA vs QLoRA），对齐原论文口径。

度量与原文 (PiSSA, NeurIPS 2024, Table 4) 一致：
  量化误差降低率 = (1 - ||W_res - nf4(W_res)||_* / ||W - nf4(W)||_*) * 100%
其中 ||·||_* 为**核范数**（nuclear / 奇异值之和）。按投影类型 (Q/K/V/O/Gate/Up/Down)
分组、跨全部层求平均，AVG = 各投影类型平均值的均值（与原文表头一致）。

同时保留 Frobenius 范数下的降低率作参考（旧口径，便于对照历史结果）。

用法（需基座模型权重 + CUDA；纯前向计算，不训练）：
    python scripts/m3/table4.py --config configs/m3_llama2_7b.yaml          # 全部层（对齐原文）
    python scripts/m3/table4.py --config configs/m3_llama2_7b.yaml --layers 2  # 冒烟（前 2 层）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def nuclear_norm(m: torch.Tensor) -> float:
    """核范数 ||M||_* = 奇异值之和（对齐原文量化误差度量）。"""
    return torch.linalg.svdvals(m.to(torch.float32)).sum().item()


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
        proj = name.rsplit(".", 1)[-1]  # 投影类型：q_proj/k_proj/.../down_proj
        W = module.weight.data.detach().to(dtype=compute_dtype)
        try:
            # === 1. 基线：NF4 直接量化原权重 W ===
            W_q = nf4_quant_dequant(W, compute_dtype)

            # === 2. PiSSA SVD 分解，得到残差 W_res ===
            init = pissa_init(W, r=r, niter=None)
            err_svd = reconstruction_error_against(init, W)
            W_res = init.residual.to(dtype=compute_dtype)

            # === 3. NF4 量化残差 W_res ===
            W_res_q = nf4_quant_dequant(W_res, compute_dtype)

            # === 4. 误差降低率 ===
            #   QLoRA  误差 = ||W      - nf4(W)     ||   （baseline）
            #   QPiSSA 误差 = ||W_res  - nf4(W_res) ||   （因 W-(nf4(W_res)+AB)=W_res-nf4(W_res)）
            # 原文口径用核范数；同时保留 Frobenius 作参考。
            base_nuc = nuclear_norm(W - W_q)
            pissa_nuc = nuclear_norm(W_res - W_res_q)
            red_nuc = (1.0 - pissa_nuc / max(base_nuc, 1e-12)) * 100.0

            base_fro = (W - W_q).norm().item()
            pissa_fro = (W_res - W_res_q).norm().item()
            red_fro = (1.0 - pissa_fro / max(base_fro, 1e-12)) * 100.0

            results.append({
                "layer": name,
                "proj_type": proj,
                "in_features": module.in_features,
                "out_features": module.out_features,
                # 原文口径（核范数）
                "baseline_err_nuclear": round(base_nuc, 6),
                "pissa_err_nuclear": round(pissa_nuc, 6),
                "reduction_pct_nuclear": round(red_nuc, 4),
                # 参考口径（Frobenius，旧结果）
                "baseline_err_frobenius": round(base_fro, 6),
                "pissa_err_frobenius": round(pissa_fro, 6),
                "reduction_pct_frobenius": round(red_fro, 4),
                "svd_recon_err": round(err_svd, 8),
            })

            print(f"  {name}: reduction(nuc)={red_nuc:5.1f}%  "
                  f"reduction(fro)={red_fro:5.1f}%  svd_err={err_svd:.2e}")

        except Exception as e:
            print(f"  {name}: SKIP ({e})")
            results.append({"layer": name, "proj_type": proj, "error": str(e)})

    # ---- 汇总：按投影类型分组（对齐原文 Table 4）----
    valid = [r for r in results if "reduction_pct_nuclear" in r]
    summary = {"error": "no valid results"}
    if valid:
        by_type_nuc: dict[str, list[float]] = defaultdict(list)
        by_type_fro: dict[str, list[float]] = defaultdict(list)
        for rec in valid:  # 不能用 r 作循环变量：会覆盖上面的 r = cfg["adapter"]["r"]
            by_type_nuc[rec["proj_type"]].append(rec["reduction_pct_nuclear"])
            by_type_fro[rec["proj_type"]].append(rec["reduction_pct_frobenius"])

        # 原文表头顺序
        order = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
        types = [t for t in order if t in by_type_nuc] + \
                sorted(t for t in by_type_nuc if t not in order)

        per_type_nuc = {t: round(sum(by_type_nuc[t]) / len(by_type_nuc[t]), 2) for t in types}
        per_type_fro = {t: round(sum(by_type_fro[t]) / len(by_type_fro[t]), 2) for t in types}
        # AVG = 各投影类型平均值的均值（与原文一致）
        avg_nuc = round(sum(per_type_nuc.values()) / len(per_type_nuc), 2)
        avg_fro = round(sum(per_type_fro.values()) / len(per_type_fro), 2)

        summary = {
            "metric": "quantization error reduction ratio (%) vs NF4(W) baseline",
            "norm": "nuclear (paper) + frobenius (reference)",
            "num_layers": len(valid),
            "reduction_pct_nuclear_by_type": per_type_nuc,
            "reduction_pct_nuclear_AVG": avg_nuc,
            "reduction_pct_frobenius_by_type": per_type_fro,
            "reduction_pct_frobenius_AVG": avg_fro,
        }

        # 终端打印一个类原文 Table 4 的小表
        print("\n=== Table 4（量化误差降低率 %，对齐原文核范数口径）===")
        header = "  " + "".join(f"{t.replace('_proj',''):>8}" for t in types) + f"{'AVG':>8}"
        print(header)
        row_nuc = "  " + "".join(f"{per_type_nuc[t]:>8.1f}" for t in types) + f"{avg_nuc:>8.1f}"
        print(f"nuclear{row_nuc}")
        row_fro = "  " + "".join(f"{per_type_fro[t]:>8.1f}" for t in types) + f"{avg_fro:>8.1f}"
        print(f"froben.{row_fro}")

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
