"""[M3] Table 4：量化重建误差对比（QPiSSA vs QLoRA），对齐原论文口径。

度量与原文 (PiSSA, NeurIPS 2024, Table 4) 一致：
  量化误差降低率 = (1 - ||W - Q_T - B_T A_T||_* / ||W - nf4(W)||_*) * 100%
其中 ||·||_* 为核范数（奇异值之和）。QPiSSA 项等于 ``||R_T - Q_T||_*``。
按投影类型分组、跨层求平均，AVG = 各投影类型平均值的均值。同时保留 Frobenius 口径作参考。

协议（HANDOFF P6/P7）：
  - FP32 计算，NF4、blocksize 64，初始化阶段关闭 double quant（``quantization.init_double_quant``）；
  - QPiSSA 为外层交替初始化，轮数由 ``--rounds`` 或 ``adapter.init_rounds`` 给出
    （1 轮为一次 SVD 后量化残差的基线，论文 Table 8 的 10.7 对应 1 轮，论文配置为 5 轮）；
    与 Fast SVD 的 ``adapter.niter`` 无关；
  - 本脚本测的是初始化阶段的量化重建误差，不包含训练加载阶段的 BF16 + double quant 重新量化。

用法（需基座模型权重 + CUDA；纯前向计算，不训练）：
    python src/scripts/m3/table4.py --config configs/paper/m3_table4_llama2_7b.yaml --rounds 1
    python src/scripts/m3/table4.py --config configs/paper/m3_table4_llama2_7b.yaml --rounds 5
    python src/scripts/m3/table4.py --config ... --layers 2   # 冒烟：只测前 2 个匹配的 Linear 模块
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adapters.quant import nf4_roundtrip
from pissa.layer_nf4 import qpissa_alternating_init

from utils import abspath, apply_overrides, load_config
from scripts.model_ops import resolve_model_path

ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def nuclear_norm(m: torch.Tensor) -> float:
    """核范数 ||M||_* = 奇异值之和（对齐原文量化误差度量）。"""
    return torch.linalg.svdvals(m.to(torch.float32)).sum().item()


def linear_layers(model: nn.Module, target_modules: tuple[str, ...]) -> list[tuple[str, nn.Linear]]:
    """按目标模块叶名称收集 nn.Linear，返回 (完整名称, 模块) 列表。"""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in target_modules]


def reduction_pct(base_err: float, qpissa_err: float) -> tuple[float | None, str]:
    """降低率 (1 - qpissa/base)*100 及状态。

    base_err == 0 时相对降低率未定义（不再用 1e-12 夹紧）；任一误差非有限则无效；
    base>0 且 qpissa==0 时允许有效的 100%；允许合法的负降低率。
    """
    if not (math.isfinite(base_err) and math.isfinite(qpissa_err)):
        return None, "non_finite"
    if base_err == 0.0:
        return None, "undefined_zero_baseline"
    return (1.0 - qpissa_err / base_err) * 100.0, "ok"


def summarize(valid: list[dict]) -> dict:
    """用未舍入的值做模块 → 类型平均 → 类型等权 AVG 汇总。"""
    by_nuc: dict[str, list[float]] = defaultdict(list)
    by_fro: dict[str, list[float]] = defaultdict(list)
    for rec in valid:
        if rec["reduction_pct_nuclear"] is not None:
            by_nuc[rec["proj_type"]].append(rec["reduction_pct_nuclear"])
        if rec["reduction_pct_frobenius"] is not None:
            by_fro[rec["proj_type"]].append(rec["reduction_pct_frobenius"])
    types = [t for t in ORDER if t in by_nuc] + sorted(t for t in by_nuc if t not in ORDER)
    per_nuc = {t: sum(by_nuc[t]) / len(by_nuc[t]) for t in types}
    per_fro = {t: sum(by_fro[t]) / len(by_fro[t]) for t in types if by_fro.get(t)}
    return {
        "reduction_pct_nuclear_by_type": per_nuc,
        "reduction_pct_nuclear_AVG": sum(per_nuc.values()) / len(per_nuc) if per_nuc else None,
        "reduction_pct_frobenius_by_type": per_fro,
        "reduction_pct_frobenius_AVG": sum(per_fro.values()) / len(per_fro) if per_fro else None,
    }


def main() -> None:
    """加载基座并计算逐投影类型的 NF4 误差统计，保存 Table 4 指标；不执行训练。"""
    parser = argparse.ArgumentParser(description="[M3] Table 4 量化误差")
    parser.add_argument("--config", default="configs/paper/m3_table4_llama2_7b.yaml")
    parser.add_argument("--rounds", type=int, default=None, help="QPiSSA 外层交替轮数；缺省取 adapter.init_rounds")
    parser.add_argument("--layers", type=int, default=None,
                        help="只测前 N 个匹配的 Linear 模块（按模块数量，不是 transformer block 数；冒烟用）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名产物")
    args = parser.parse_args()
    if args.layers is not None and args.layers <= 0:
        raise SystemExit("--layers 必须是正整数。")

    cfg = load_config(abspath(args.config))
    apply_overrides(cfg, args.set)
    acfg, qcfg = cfg["adapter"], cfg["quantization"]
    rounds = args.rounds if args.rounds is not None else acfg.get("init_rounds", 1)
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise SystemExit(f"轮数必须是 >=1 的整数，收到 {rounds!r}")
    init_dq = bool(qcfg.get("init_double_quant", False))
    niter = acfg.get("niter")

    if not torch.cuda.is_available():
        raise SystemExit("Table 4 需要 CUDA（bitsandbytes NF4）。")
    device = "cuda"
    out_dir = abspath(cfg["paths"]["output_dir"])
    scope = "full" if args.layers is None else f"layers{args.layers}"
    out_path = out_dir / f"table4_quant_error_rounds{rounds}_{scope}.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"已存在 {out_path}；请显式传 --overwrite。")

    from transformers import AutoModelForCausalLM

    model_path = resolve_model_path(cfg)
    print(f"加载模型 {model_path}（FP32，不含量化）...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    model.eval()

    target_modules = tuple(acfg["target_modules"])
    r = acfg["r"]
    lines = linear_layers(model, target_modules)
    n_matched = len(lines)
    if args.layers:
        lines = lines[: args.layers]
    expected_by_type: dict[str, int] = defaultdict(int)
    for name, _ in lines:
        expected_by_type[name.rsplit(".", 1)[-1]] += 1
    print(f"匹配 {n_matched} 个线性模块，本次测量 {len(lines)} 个；rounds={rounds} niter={niter} init_double_quant={init_dq}")

    records: list[dict] = []
    for name, module in lines:
        proj = name.rsplit(".", 1)[-1]
        try:
            W = module.weight.data.detach().to(torch.float32)
            W_q = nf4_roundtrip(W, compress_statistics=init_dq)  # QLoRA 基线：NF4(W)
            init = qpissa_alternating_init(W, r=r, rounds=rounds, niter=niter, double_quant=init_dq)
            E = W - (init.quantized_residual + init.lora_B @ init.lora_A)  # = R_T - Q_T
            base_nuc, qp_nuc = nuclear_norm(W - W_q), nuclear_norm(E)
            base_fro, qp_fro = (W - W_q).norm().item(), E.norm().item()
            red_nuc, st_nuc = reduction_pct(base_nuc, qp_nuc)
            red_fro, st_fro = reduction_pct(base_fro, qp_fro)
            records.append({
                "layer": name, "proj_type": proj,
                "in_features": module.in_features, "out_features": module.out_features,
                "baseline_err_nuclear": base_nuc, "qpissa_err_nuclear": qp_nuc,
                "reduction_pct_nuclear": red_nuc, "status_nuclear": st_nuc,
                "baseline_err_frobenius": base_fro, "qpissa_err_frobenius": qp_fro,
                "reduction_pct_frobenius": red_fro, "status_frobenius": st_fro,
                "round_errors_rel_fro": init.round_errors,
            })
            shown = f"{red_nuc:5.1f}%" if red_nuc is not None else st_nuc
            print(f"  {name}: reduction(nuc)={shown}")
        except Exception as e:  # noqa: BLE001 - 失败逐条记录，由状态汇总暴露
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")
            records.append({"layer": name, "proj_type": proj, "status": "failed",
                            "error": f"{type(e).__name__}: {e}"})

    failed = [rec for rec in records if rec.get("status") == "failed"]
    valid = [rec for rec in records if "status" not in rec and rec["reduction_pct_nuclear"] is not None]
    excluded = [rec for rec in records if "status" not in rec and rec["reduction_pct_nuclear"] is None]
    ok_by_type: dict[str, int] = defaultdict(int)
    for rec in valid:
        ok_by_type[rec["proj_type"]] += 1
    covers_all = len(valid) == len(lines) and all(ok_by_type[t] == n for t, n in expected_by_type.items())
    status = "complete" if covers_all else ("failed" if not valid else "partial")
    summary = summarize(valid) if valid else {}
    summary.update({
        "metric": "quantization error reduction ratio (%) vs NF4(W) baseline",
        "norm": "nuclear (paper) + frobenius (reference)",
        "expected_modules": len(lines), "valid_modules": len(valid),
        "failed_modules": len(failed), "excluded_modules": len(excluded),
        "expected_by_type": dict(expected_by_type), "valid_by_type": dict(ok_by_type),
        "AVG_covers_all_expected_modules": covers_all,
    })
    if valid:
        print("\n=== Table 4（量化误差降低率 %，核范数口径）===  状态:", status)
        types = list(summary["reduction_pct_nuclear_by_type"])
        print("  " + "".join(f"{t.replace('_proj', ''):>8}" for t in types) + f"{'AVG':>8}")
        print("nuc  " + "".join(f"{summary['reduction_pct_nuclear_by_type'][t]:>8.1f}" for t in types)
              + f"{summary['reduction_pct_nuclear_AVG']:>8.1f}")
        if not covers_all:
            print("  [warn] AVG 未覆盖全部预期模块，不能当作完整结果。")

    from scripts.m1.train import code_version, env_info

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "M3/Table4 量化重建误差",
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "r": r, "target_modules": list(target_modules), "quant_type": "nf4", "blocksize": 64,
            "init_double_quant": init_dq, "qpissa_rounds": rounds, "svd_niter": niter,
            "measurement_precision": "float32", "scope": scope, "num_matched_modules": n_matched,
            "config_path": str(abspath(args.config)), "overrides": args.set,
            "code": code_version(), "env": env_info(),
        },
        "summary": summary,
        "per_layer": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {out_path}")
    if status != "complete":
        raise SystemExit(f"Table 4 状态={status}（失败 {len(failed)}，排除 {len(excluded)}）。")


if __name__ == "__main__":
    main()
