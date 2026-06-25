"""[M0] 验证 PiSSA 从零实现的正确性。

两件事：
  1. 重构恒等：W_res + B @ A ≈ W（相对误差应接近浮点精度）；
     且 PiSSALinear 在初始化时的前向输出与原始 nn.Linear 一致。
  2. 与 HuggingFace PEFT 内置 PiSSA 数值对齐（若已安装 peft/transformers）。

用法：
    python scripts/m0/verify_init.py --in-features 1024 --out-features 1024 --r 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 让脚本能直接 import 到 src/pissa（无需安装为包）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pissa import PiSSALinear, pissa_init, reconstruction_error_against  # noqa: E402


def check_reconstruction(in_f: int, out_f: int, r: int, niter: int | None) -> None:
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


def check_against_peft(r: int) -> None:
    print("== 2. 与 PEFT 内置 PiSSA 对齐 ==")
    try:
        import transformers  # noqa: F401
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - 依赖缺失时优雅跳过
        print(f"  跳过：未安装 peft/transformers（{exc}）\n")
        return

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
    print(f"  主成分 B@A 相对误差   = {rel(ours_delta, peft_delta):.3e}")
    print(f"  残差 W_res 相对误差    = {rel(ours.residual.float(), peft_res.float()):.3e}")
    ok = rel(ours_delta, peft_delta) < 1e-4 and rel(ours.residual.float(), peft_res.float()) < 1e-4
    print(f"  结论：{'对齐' if ok else '不一致，需排查'}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="[M0] 验证 PiSSA 初始化正确性")
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--niter", type=int, default=None, help="设置则用快速随机化 SVD")
    parser.add_argument("--skip-peft", action="store_true")
    args = parser.parse_args()

    check_reconstruction(args.in_features, args.out_features, args.r, args.niter)
    if not args.skip_peft:
        check_against_peft(args.r)


if __name__ == "__main__":
    main()
