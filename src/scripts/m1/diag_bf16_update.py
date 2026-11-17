"""[M1 诊断] BF16 权重 + AdamW 的“小更新被舍入”检测（不训练、不保存模型、不碰 GSM8K）。

问题：Full-FT 按论文用 BF16 参数。BF16 只有 8 位有效数字，当某个元素的 AdamW 更新量
``lr * m_hat / (sqrt(v_hat) + eps)`` 小于该权重 BF16 间隔的一半时，``w - update`` 舍入回 ``w``，更新丢失。
官方 DeepSpeed 是否带 FP32 主权重未核实，本脚本只度量“纯 BF16 权重”下的丢失比例。

方法：
  1. 按训练的同一套数据/分词/窗口/损失加权，用 BF16 模型对前 K 个真实梯度窗口累积 FP32 的 Adam
     一阶/二阶矩（权重保持不变，仅估计更新量的分布）；
  2. 对若干个学习率水平（峰值，以及按真实调度在不同训练进度处的取值），逐元素计算
     ``new = (w.float() - lr * r).to(bf16)``，统计 ``new == w`` 且 ``r != 0`` 的比例（丢失比例），
     并按参数类型分组，同时给出“被保留的更新量占比”。

局限：K 步之后的矩估计偏向训练早期（v_hat 只平均了 K 个窗口），晚期 |r| 通常更小，丢失比例会更高；
这里的 r 只代表早期梯度噪声水平。结果是诊断依据，不是对最终准确率的预测。

显存：BF16 模型 + BF16 梯度 + FP32 的 m、v，7B 约 81GB，需 96GB 级显卡；不足时可加
``--set methods.full.per_device_batch=2`` 等，但 m、v 本身固定占用，OOM 时请改用 ``--state-dtype bfloat16``（会改变估计精度）。

用法：
    python src/scripts/m1/diag_bf16_update.py --config configs/paper/m1_llama2_7b.yaml --windows 4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from scripts.data_ops import Collator, read_jsonl
from scripts.m1.train import (
    build_examples,
    code_version,
    env_info,
    hf_cosine_lr,
    iter_windows,
    plan_updates,
    resolve_autocast,
    resolve_sample_count,
    valid_token_count,
    validate_cfg,
    warmup_steps_for,
)
from scripts.model_ops import fix_unregistered_meta_tensors, load_base_model, load_tokenizer, resolve_model_path
from utils import abspath, apply_overrides, load_config, set_seed, torch_dtype

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8  # torch.optim.AdamW 默认值（官方 adamw_torch 同）


def param_group(name: str) -> str:
    """把参数名归入诊断分组。"""
    leaf = name.rsplit(".", 2)
    if "embed_tokens" in name:
        return "embed_tokens"
    if "lm_head" in name:
        return "lm_head"
    if "layernorm" in name or name.endswith("norm.weight"):
        return "norm"
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if proj in name:
            return proj
    return "other:" + ".".join(leaf[-2:])


def update_lost(w: torch.Tensor, r: torch.Tensor, lr: float) -> tuple[int, float, float]:
    """对 BF16 权重 w 施加更新 ``-lr*r``（在 FP32 中计算再舍入回 BF16），统计丢失情况。

    Returns:
        (r 非零且舍入后权重不变的元素数, 期望更新量绝对值之和, 实际应用的更新量绝对值之和)。
    """
    wf = w.float()
    u = lr * r
    new = (wf - u).to(torch.bfloat16)
    lost = (new == w) & (r != 0)
    return int(lost.sum()), float(u.abs().sum()), float((new.float() - wf).abs().sum())


def lr_levels(base_lr: float, total_steps: int, warmup: int) -> dict[str, float]:
    """按真实余弦调度取若干代表性学习率：峰值与训练进度 25/50/75/90/99% 处。"""
    levels = {"peak": base_lr}
    for frac in (0.25, 0.5, 0.75, 0.9, 0.99):
        step = max(warmup, min(total_steps - 1, int(frac * total_steps)))
        levels[f"progress_{int(frac * 100)}%"] = hf_cosine_lr(step, total_steps, warmup, base_lr)
    return levels


def main() -> None:
    """加载 BF16 模型，累积 K 个真实窗口的 Adam 矩，统计各学习率水平下的更新丢失比例并保存 JSON。"""
    parser = argparse.ArgumentParser(description="[M1 诊断] BF16 + AdamW 小更新检测")
    parser.add_argument("--config", default="configs/paper/m1_llama2_7b.yaml")
    parser.add_argument("--windows", type=int, default=4, help="用于估计 Adam 矩的真实梯度窗口数 K")
    parser.add_argument("--limit", type=int, default=None, help="分词样本上限；缺省取 windows*有效batch*2")
    parser.add_argument("--state-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.windows < 1:
        raise SystemExit("--windows 必须 >=1")
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA。")

    cfg = load_config(abspath(args.config))
    apply_overrides(cfg, args.set)
    validate_cfg(cfg, "full")
    mcfg, tcfg, ocfg = cfg["methods"]["full"], cfg["train"], cfg["optim"]
    if mcfg["dtype"] != "bfloat16":
        raise SystemExit(f"本诊断针对 BF16 参数，methods.full.dtype={mcfg['dtype']!r}")
    out_path = abspath(cfg["paths"]["output_dir"]) / "metrics" / "diag_bf16_update.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"已存在 {out_path}；请显式传 --overwrite。")

    set_seed(tcfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = bool(tcfg.get("tf32", True))
    per_device, grad_accum = mcfg["per_device_batch"], mcfg["grad_accum"]
    window_samples = per_device * grad_accum
    limit = args.limit if args.limit is not None else args.windows * window_samples * 2

    model_path = resolve_model_path(cfg)
    tokenizer = load_tokenizer(model_path)
    built = build_examples(cfg, tokenizer, limit)
    examples = built["examples"]
    if len(examples) // per_device // grad_accum < args.windows:
        raise SystemExit(f"样本 {len(examples)} 条不足 {args.windows} 个窗口，请增大 --limit。")

    # 学习率水平按“完整配置”的总步数计算（不是本次诊断的小样本）。
    n_rows = len(read_jsonl(abspath(cfg["data"]["train_jsonl"])))
    n_full, _ = resolve_sample_count(cfg["data"].get("num_train_samples"), None, n_rows)
    total_steps = plan_updates(n_full, per_device, grad_accum, tcfg["num_epochs"])["total_opt_steps"]
    warmup = warmup_steps_for(total_steps, ocfg["warmup_ratio"])
    base_lr = mcfg.get("learning_rate") or ocfg["learning_rate"]
    levels = lr_levels(base_lr, total_steps, warmup)
    print(f"完整配置总更新步={total_steps} warmup={warmup}；学习率水平：{levels}")

    dtype = torch_dtype(mcfg["dtype"])
    amp_dtype = resolve_autocast(mcfg, dtype)
    model = load_base_model(model_path, dtype, low_cpu_mem_usage=cfg.get("model", {}).get("low_cpu_mem_usage", True))
    model.to("cuda")
    fix_unregistered_meta_tensors(model, "cuda")
    if mcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    params = [(n, p) for n, p in model.named_parameters()]

    gen = torch.Generator()
    gen.manual_seed(int(tcfg["seed"]))
    loader = DataLoader(examples, batch_size=per_device, shuffle=True, generator=gen,
                        collate_fn=Collator(tokenizer.pad_token_id), drop_last=True)
    sdt = torch_dtype(args.state_dtype)
    m_state = {n: torch.zeros_like(p, dtype=sdt) for n, p in params}
    v_state = {n: torch.zeros_like(p, dtype=sdt) for n, p in params}

    t0 = time.time()
    for k, window in enumerate(iter_windows(loader, grad_accum)):
        if k >= args.windows:
            break
        counts = [valid_token_count(b["labels"]) for b in window]
        wt = sum(counts)
        for batch, n_tok in zip(window, counts):
            if n_tok == 0:
                continue
            batch = {a: b.to("cuda") for a, b in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else _null():
                loss = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"]).loss.float()
            (loss * (n_tok / wt)).backward()
        gn = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)  # 与训练一致的梯度裁剪
        if not torch.isfinite(gn):
            raise RuntimeError(f"窗口 {k} 梯度范数非有限：{float(gn)}")
        for n, p in params:
            g = p.grad.float()
            m_state[n].mul_(BETA1).add_(g.to(sdt), alpha=1 - BETA1)
            v_state[n].mul_(BETA2).addcmul_(g.to(sdt), g.to(sdt), value=1 - BETA2)
            p.grad = None
        print(f"  窗口 {k + 1}/{args.windows} 完成，grad_norm(裁剪前)={float(gn):.4f}，用时 {time.time() - t0:.0f}s")

    K = args.windows
    bc1, bc2 = 1 - BETA1 ** K, 1 - BETA2 ** K
    acc: dict[str, dict] = {}
    for n, p in params:
        grp = param_group(n)
        d = acc.setdefault(grp, {"n": 0, "nonzero_r": 0, "abs_r_sum": 0.0, "small_r_lt_0.1": 0,
                                 "levels": {lv: {"lost": 0, "u_sum": 0.0, "applied_sum": 0.0} for lv in levels}})
        w = p.data
        r = (m_state[n].float() / bc1) / ((v_state[n].float() / bc2).sqrt() + EPS)
        nz = r != 0
        d["n"] += r.numel()
        d["nonzero_r"] += int(nz.sum())
        d["abs_r_sum"] += float(r.abs().sum())
        d["small_r_lt_0.1"] += int((r.abs() < 0.1).sum())
        for lv, lr in levels.items():
            lost, u_sum, applied_sum = update_lost(w, r, lr)
            e = d["levels"][lv]
            e["lost"] += lost
            e["u_sum"] += u_sum
            e["applied_sum"] += applied_sum
        del r
    # 汇总
    def summarize(d):
        out = {"num_elements": d["n"], "nonzero_update_fraction": d["nonzero_r"] / d["n"],
               "mean_abs_r": d["abs_r_sum"] / d["n"], "fraction_abs_r_lt_0.1": d["small_r_lt_0.1"] / d["n"], "levels": {}}
        for lv, e in d["levels"].items():
            out["levels"][lv] = {
                "lr": levels[lv],
                "lost_fraction_of_nonzero": e["lost"] / max(d["nonzero_r"], 1),
                "applied_over_intended_magnitude": e["applied_sum"] / e["u_sum"] if e["u_sum"] > 0 else None,
            }
        return out

    total = {"n": 0, "nonzero_r": 0, "abs_r_sum": 0.0, "small_r_lt_0.1": 0,
             "levels": {lv: {"lost": 0, "u_sum": 0.0, "applied_sum": 0.0} for lv in levels}}
    for d in acc.values():
        for key in ("n", "nonzero_r", "abs_r_sum", "small_r_lt_0.1"):
            total[key] += d[key]
        for lv in levels:
            for key in ("lost", "u_sum", "applied_sum"):
                total["levels"][lv][key] += d["levels"][lv][key]
    results = {"overall": summarize(total), "by_group": {g: summarize(d) for g, d in sorted(acc.items())}}

    print("\n=== 更新丢失比例（nonzero 更新中被 BF16 舍入掉的比例）===")
    print(f"{'group':14}" + "".join(f"{lv:>14}" for lv in levels))
    for g, s in [("OVERALL", results["overall"])] + list(results["by_group"].items()):
        print(f"{g:14}" + "".join(f"{s['levels'][lv]['lost_fraction_of_nonzero']:>14.3f}" for lv in levels))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "experiment": "M1/diag_bf16_update",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"config_path": str(abspath(args.config)), "overrides": args.set, "windows_K": K,
                   "adam": {"beta1": BETA1, "beta2": BETA2, "eps": EPS, "weight_decay": 0.0},
                   "state_dtype": args.state_dtype, "effective_batch": window_samples,
                   "num_samples_tokenized": len(examples), "full_run_total_steps": total_steps,
                   "full_run_warmup_steps": warmup, "lr_levels": levels, "max_seq_len": tcfg["max_seq_len"],
                   "code": code_version(), "env": env_info()},
        "notes": [
            "权重保持初始值不变，仅用前 K 个真实窗口估计 Adam 矩；K 步后的 v_hat 偏向训练早期。",
            "FP32 主权重下丢失比例按定义为 0；官方 DeepSpeed 是否使用主权重未核实。",
            "该指标只度量‘更新被舍入掉’，不预测最终准确率。",
        ],
        "results": results,
        "wall_sec": time.time() - t0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {out_path}")


class _null:
    """不需要 autocast 时的空上下文。"""
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
