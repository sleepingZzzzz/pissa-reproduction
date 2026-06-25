"""[M0] 验证 PiSSA 从零实现的正确性（多根互相独立的支柱，防止"两实现一起错"）。

三件事：
  1. 重构恒等：W_res + B @ A ≈ W（相对误差应接近浮点精度）；
     且 PiSSALinear 在初始化时的前向输出与原始 nn.Linear 一致。
     注：因 W_res 由 W - B@A 定义，本项只验证算术/接线，不验证主成分取法。
  2. SVD 数学性质（Eckart–Young）：||W_res||_F = sqrt(Σ_{i>r} σ_i²)、||W_res||_2 = σ_{r+1}，
     独立于任何外部实现地验证 B@A 确为前 r 个主成分。
  3. 与 HuggingFace PEFT 内置 PiSSA 数值对齐（若已安装 peft/transformers）。

用法：
    python scripts/m0/verify_init.py --in-features 1024 --out-features 1024 --r 64
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# 让脚本能直接 import 到 src/pissa（无需安装为包）。
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pissa import PiSSALinear, pissa_init, reconstruction_error_against  # noqa: E402

METRICS_DIR = REPO_ROOT / "results" / "m0" / "exp1" / "metrics"


def check_reconstruction(in_f: int, out_f: int, r: int, niter: int | None) -> dict:
    print("== 1. 重构恒等 & 前向一致性 ==")
    torch.manual_seed(0)
    linear = nn.Linear(in_f, out_f, bias=True)

    init = pissa_init(linear.weight.data, r=r, niter=niter)
    rel_err = reconstruction_error_against(init, linear.weight.data)
    print(f"  ||W_res + B@A - W||_F / ||W||_F = {rel_err:.3e}")

    pissa_layer = PiSSALinear.from_linear(linear, r=r, alpha=r, niter=niter)
    x = torch.randn(8, in_f)
    with torch.no_grad():
        y_ref = linear(x)
        y_pissa = pissa_layer(x)
    fwd_err = float(torch.linalg.norm(y_pissa - y_ref) / torch.linalg.norm(y_ref).clamp_min(1e-12))
    print(f"  初始化前向相对误差 = {fwd_err:.3e}")

    tol = 1e-4 if niter is not None else 1e-5
    ok = rel_err < tol and fwd_err < tol
    print(f"  结论：{'通过' if ok else '未通过'}（容差 {tol:.0e}）\n")
    return {
        "reconstruction_rel_err": rel_err,
        "forward_rel_err": fwd_err,
        "tol": tol,
        "passed": bool(ok),
    }


def check_svd_property(in_f: int, out_f: int, r: int) -> dict:
    """支柱②：用 SVD 的 Eckart–Young 性质独立验证 B@A 确为前 r 个主成分。

    不依赖任何外部实现。最优 rank-r 近似的残差满足：
        ||W - B@A||_F = sqrt(Σ_{i>r} σ_i²)，  ||W - B@A||_2 = σ_{r+1}
    残差由矩阵减法 W - B@A 得到，右端由奇异值列表得到，二者本是独立来源；
    若主成分取错（例如取了最小的 r 个、或拼装错位），等式立即不成立。
    """
    print("== 2. SVD 数学性质（Eckart–Young，独立于 PEFT） ==")
    torch.manual_seed(0)
    W = torch.randn(out_f, in_f)
    init = pissa_init(W, r=r)  # 走精确 SVD

    residual = init.residual.float()
    S = init.singular_values.float()  # 全部奇异值，降序

    # 残差 Frobenius 范数应等于被丢弃奇异值的平方和开根。
    fro_lhs = float(torch.linalg.norm(residual))
    fro_rhs = float(torch.sqrt((S[r:] ** 2).sum()))
    fro_err = abs(fro_lhs - fro_rhs) / max(fro_rhs, 1e-12)

    # 残差谱顶（最大奇异值）应等于原矩阵第 r+1 个奇异值（0-indexed S[r]）。
    spec_lhs = float(torch.linalg.matrix_norm(residual, ord=2))
    spec_rhs = float(S[r]) if r < S.numel() else 0.0
    spec_err = abs(spec_lhs - spec_rhs) / max(spec_rhs, 1e-12)

    # float32 对 O(k) 个奇异值平方求和会有累积误差；容差按 float32 现实设定，
    # 仍远紧于"取错主成分"会产生的 O(1) 误差，足以判别正确性。
    fro_tol, spec_tol = 1e-3, 1e-4
    print(f"  ||W_res||_F vs sqrt(sum_(i>r) sigma_i^2) = {fro_lhs:.4f} vs {fro_rhs:.4f}  (相对误差 {fro_err:.3e})")
    print(f"  ||W_res||_2 vs sigma_(r+1)               = {spec_lhs:.4f} vs {spec_rhs:.4f}  (相对误差 {spec_err:.3e})")
    ok = fro_err < fro_tol and spec_err < spec_tol
    print(f"  结论：{'通过（B@A 确为前 r 个主成分）' if ok else '未通过，主成分取法可疑'}（容差 F={fro_tol:.0e}, 谱={spec_tol:.0e}）\n")
    return {
        "frobenius_lhs": fro_lhs,
        "frobenius_rhs": fro_rhs,
        "frobenius_rel_err": fro_err,
        "frobenius_tol": fro_tol,
        "spectral_lhs": spec_lhs,
        "spectral_rhs": spec_rhs,
        "spectral_rel_err": spec_err,
        "spectral_tol": spec_tol,
        "passed": bool(ok),
    }


def check_against_peft(r: int) -> dict:
    print("== 3. 与 PEFT 内置 PiSSA 对齐 ==")
    try:
        import transformers  # noqa: F401
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - 依赖缺失时优雅跳过
        print(f"  跳过：未安装 peft/transformers（{exc}）\n")
        return {"skipped": True, "reason": f"import failed: {exc}"}

    # 构造一个极小的随机 Llama 以便快速对齐，无需联网下载权重。
    cfg = AutoConfig.for_model(
        "llama",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=320,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.float32)

    # 取第 0 层 q_proj 作为对照样本。
    target_name = "model.layers.0.self_attn.q_proj"
    base_weight = model.get_submodule(target_name).weight.data.clone()

    # 我们的实现
    ours = pissa_init(base_weight, r=r)

    # PEFT 的实现（init_lora_weights="pissa"，alpha=r → scaling=1）
    lconf = LoraConfig(
        r=r,
        lora_alpha=r,
        target_modules=["q_proj"],
        init_lora_weights="pissa",
        lora_dropout=0.0,
    )
    peft_model = get_peft_model(model, lconf)
    qproj = peft_model.get_submodule("base_model.model." + target_name)
    peft_A = qproj.lora_A["default"].weight.data
    peft_B = qproj.lora_B["default"].weight.data
    peft_res = qproj.base_layer.weight.data

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp_min(1e-12))

    # A、B 的分解不唯一（可差一个 r×r 正交/符号变换），但 B@A 与残差必须一致。
    ours_delta = ours.lora_B.float() @ ours.lora_A.float()
    peft_delta = peft_B.float() @ peft_A.float()
    ba_err = rel(ours_delta, peft_delta)
    res_err = rel(ours.residual.float(), peft_res.float())
    print(f"  主成分 B@A 相对误差   = {ba_err:.3e}")
    print(f"  残差 W_res 相对误差    = {res_err:.3e}")
    ok = ba_err < 1e-4 and res_err < 1e-4
    print(f"  结论：{'对齐' if ok else '不一致，需排查'}\n")
    return {
        "skipped": False,
        "reference_layer": target_name,
        "peft_BA_rel_err": ba_err,
        "peft_res_rel_err": res_err,
        "passed": bool(ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="[M0] 验证 PiSSA 初始化正确性")
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--niter", type=int, default=None, help="设置则用快速随机化 SVD")
    parser.add_argument("--skip-peft", action="store_true")
    args = parser.parse_args()

    config = {
        "in_features": args.in_features,
        "out_features": args.out_features,
        "r": args.r,
        "niter": args.niter,
        "skip_peft": bool(args.skip_peft),
        "seed": 0,
    }
    results = {
        "reconstruction_and_forward": check_reconstruction(
            args.in_features, args.out_features, args.r, args.niter
        ),
        "svd_property_eckart_young": check_svd_property(
            args.in_features, args.out_features, args.r
        ),
        "peft_alignment": (
            check_against_peft(args.r)
            if not args.skip_peft
            else {"skipped": True, "reason": "--skip-peft"}
        ),
    }
    all_pass = all(
        v.get("passed", False)
        for v in results.values()
        if not v.get("skipped", False)
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "verify.json"
    payload = {
        "experiment": "M0/verify_init",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "results": results,
        "all_pass": all_pass,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 metrics：{out_path}")


if __name__ == "__main__":
    main()
