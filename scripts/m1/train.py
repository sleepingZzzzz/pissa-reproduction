"""[M1] 训练：用 我们的 PiSSA / LoRA / Full-FT 在 MetaMathQA 上微调 LLaMA-2-7B。

每种方法的精度/优化器/显存策略由 config 的 methods.<method> 决定。
训练后把适配器合并回权重并 save_pretrained，供 eval_gsm8k.py 评测。

用法：
    python scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method pissa
    python scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method lora  --limit 64   # 冒烟
    python scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method full
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    Collator,
    abspath,
    apply_adapter,
    build_optimizer,
    load_base_model,
    load_config,
    load_tokenizer,
    merge_adapter,
    read_jsonl,
    resolve_model_path,
    set_seed,
    tokenize_example,
    torch_dtype,
)


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description="[M1] 训练 PiSSA/LoRA/Full")
    parser.add_argument("--config", default="configs/m1_llama2_7b.yaml")
    parser.add_argument("--method", required=True, choices=["pissa", "lora", "full"])
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 条训练（冒烟）")
    args = parser.parse_args()

    cfg = load_config(abspath(args.config))
    mcfg = cfg["methods"][args.method]
    tcfg = cfg["train"]
    ocfg = cfg["optim"]
    set_seed(tcfg["seed"])

    # 开启 TF32：在 Ampere+ 卡上大幅加速 fp32 矩阵乘，数值上几乎无损，不影响与论文对齐的结论。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch_dtype(mcfg["dtype"])
    print(f"方法={args.method} 设备={device} dtype={mcfg['dtype']}")

    # ---- 模型 / 分词器 ----
    model_path = resolve_model_path(cfg)
    tokenizer = load_tokenizer(model_path)
    model = load_base_model(model_path, dtype)
    model.to(device)
    if mcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    stats = apply_adapter(model, args.method, cfg)
    print(
        f"  替换层数={stats['replaced_modules']}  可训练参数={stats['trainable_params']:,}"
        f"  占比={stats['trainable_pct']:.4f}%"
    )

    # ---- 数据 ----
    rows = read_jsonl(abspath(cfg["data"]["train_jsonl"]))
    if args.limit is not None:
        rows = rows[: args.limit]
    max_len = tcfg["max_seq_len"]
    resp_only = tcfg["loss_on_response_only"]
    examples = [
        tokenize_example(tokenizer, r["instruction"], r["response"], max_len, resp_only)
        for r in tqdm(rows, desc="分词", unit="ex")
    ]
    collator = Collator(tokenizer.pad_token_id)
    loader = DataLoader(
        examples,
        batch_size=mcfg["per_device_batch"],
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    grad_accum = mcfg["grad_accum"]
    micro_per_epoch = len(loader)
    opt_steps_per_epoch = micro_per_epoch // grad_accum
    total_opt_steps = opt_steps_per_epoch * tcfg["num_epochs"]
    warmup_steps = int(total_opt_steps * ocfg["warmup_ratio"])
    base_lr = mcfg["learning_rate"] or ocfg["learning_rate"]
    print(
        f"  样本={len(examples)} micro_bs={mcfg['per_device_batch']} grad_accum={grad_accum}"
        f" → 全局batch={mcfg['per_device_batch'] * grad_accum}; optimizer steps={total_opt_steps}"
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer, opt_used = build_optimizer(
        trainable, base_lr, ocfg["weight_decay"], mcfg["optimizer"]
    )

    use_amp = dtype in (torch.bfloat16, torch.float16)
    amp_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16

    # ---- 训练循环 ----
    model.train()
    loss_curve: list[dict] = []
    opt_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_opt_steps, desc=f"训练[{args.method}]", unit="step")
    for epoch in range(tcfg["num_epochs"]):
        for micro_i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_amp and device == "cuda":
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                ctx = _nullctx()
            with ctx:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss / grad_accum
            loss.backward()

            if (micro_i + 1) % grad_accum == 0:
                lr_now = cosine_lr(opt_step, total_opt_steps, warmup_steps, base_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                cur = float(loss.item()) * grad_accum
                pbar.update(1)
                pbar.set_postfix(loss=f"{cur:.4f}", lr=f"{lr_now:.2e}")
                if opt_step % tcfg["log_every"] == 0 or opt_step == total_opt_steps:
                    loss_curve.append({"step": opt_step, "loss": cur, "lr": lr_now})

    pbar.close()

    # ---- 合并 + 保存 ----
    merge_adapter(model, args.method)
    save_dtype = torch_dtype(tcfg["save_merged_dtype"])
    model.to(save_dtype)
    ckpt_dir = abspath(cfg["paths"]["ckpt_dir"]) / args.method
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    # transformers 5.x 会严格校验 generation_config：LLaMA-2 出厂带 temperature/top_p
    # 但 do_sample=False，直接保存会报 invalid。置 do_sample=True 使其合法（eval 仍显式贪心）。
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = True
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"已保存合并模型到 {ckpt_dir}")

    # ---- metrics ----
    metrics_dir = abspath(cfg["paths"]["output_dir"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"train_{args.method}.json"
    payload = {
        "experiment": f"M1/train ({args.method})",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "method": args.method,
            "dtype": mcfg["dtype"],
            "optimizer": opt_used,
            "grad_checkpointing": mcfg["grad_checkpointing"],
            "per_device_batch": mcfg["per_device_batch"],
            "grad_accum": grad_accum,
            "global_batch": mcfg["per_device_batch"] * grad_accum,
            "lr": base_lr,
            "num_samples": len(examples),
            "total_opt_steps": total_opt_steps,
            "max_seq_len": max_len,
            "r": cfg["adapter"]["r"],
        },
        "param_stats": stats,
        "results": {
            "loss_curve": loss_curve,
            "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
            "wall_time_sec": time.time() - t0,
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 metrics：{out_path}")


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
