"""[M1] 训练：用 我们的 PiSSA / LoRA / Full-FT 在 MetaMathQA 上微调 LLaMA-2-7B。

每种方法的精度/优化器/显存策略由 config 的 methods.<method> 决定。
训练后把适配器合并回权重并 save_pretrained，供 eval_gsm8k.py 评测。

用法：
    python src/scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method pissa
    python src/scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method lora  --limit 64   # 冒烟
    python src/scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method full
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接执行文件时，将 src 加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data_ops import Collator, read_jsonl, tokenize_example
from utils import abspath, load_config, set_seed, torch_dtype
from scripts.model_ops import (
    apply_adapter,
    build_optimizer,
    fix_unregistered_meta_tensors,
    load_base_model,
    load_tokenizer,
    merge_adapter,
    resolve_model_path,
)
from lora.inject_nf4 import unwrap_qlora


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    """返回给定优化步的学习率：先线性预热，再按余弦衰减。

    Args:
        step: 从零开始的优化步编号。
        total: 总优化步数。
        warmup: 预热步数。
        base_lr: 预热结束时的学习率。

    Returns:
        该步学习率；本函数不修改优化器。
    """
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


FINAL_LOSS_POINTS = 10  # final_loss 取最后多少个记录点的均值


def _tail_mean(curve: list[dict], key: str, n: int) -> float | None:
    """返回曲线末 n 个记录中 key 字段的算术平均；曲线为空时返回 None。"""
    tail = curve[-n:]
    return sum(c[key] for c in tail) / len(tail) if tail else None


def main() -> None:
    """执行配置驱动的微调，合并或导出适配器，保存 checkpoint 与训练指标。

    从命令行选择方法和可选样本上限；读取 JSONL、分词、右填充，按配置进行
    梯度累积、调度和优化。记录累积窗口的平均 loss，final_loss 为末尾记录均值。
    调用会加载模型、可能下载权重并训练；导入模块不会自动执行。
    """
    parser = argparse.ArgumentParser(description="[M1] 训练 PiSSA/LoRA/Full")
    parser.add_argument("--config", default="configs/m1_llama2_7b.yaml")
    parser.add_argument("--method", required=True, choices=["pissa", "lora", "full", "qpissa", "qlora"])
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
    trc = cfg.get("model", {}).get("trust_remote_code", False)
    low_cpu = cfg.get("model", {}).get("low_cpu_mem_usage", True)
    tokenizer = load_tokenizer(model_path, trust_remote_code=trc)

    # NF4 量化配置（M3 QLoRA）。QPiSSA 走 FP16 加载→PiSSA SVD→NF4 残差，不走此分支。
    quant_cfg = None
    is_qlora = args.method == "qlora"
    if is_qlora and cfg.get("quantization", {}).get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        q = cfg["quantization"]
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype(q.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        )
        model = load_base_model(model_path, quant_cfg.bnb_4bit_compute_dtype, trust_remote_code=trc,
                                low_cpu_mem_usage=False, quantization_config=quant_cfg)
        device = "cuda"
    else:
        model = load_base_model(model_path, dtype, trust_remote_code=trc, low_cpu_mem_usage=low_cpu)
        model.to(device)
    # Baichuan2 等模型的 RotaryEmbedding 缓存（cos_cached/sin_cached/inv_freq）是
    # 普通 tensor 属性（未 register_buffer），model.to() 不会移动它们；
    # low_cpu_mem_usage 的 _fast_init 使其留在 meta device → 首次 forward 必崩。
    fix_unregistered_meta_tensors(model, device)
    if mcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    stats = apply_adapter(model, args.method, cfg)
    print(
        f"  替换层数={stats['replaced_modules']}  可训练参数={stats['trainable_params']:,}"
        f"  占比={stats['trainable_pct']:.4f}%"
    )

    # ---- 数据 ----
    rows = read_jsonl(abspath(cfg["data"]["train_jsonl"]))
    # data.num_train_samples 决定实际训练规模（null = 用 jsonl 全部）；--limit 仅用于冒烟，优先级更高。
    n_cfg = cfg["data"].get("num_train_samples")
    if n_cfg is not None:
        if len(rows) < n_cfg:
            print(f"  [warn] jsonl 只有 {len(rows)} 条，少于 data.num_train_samples={n_cfg}")
        rows = rows[:n_cfg]
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
    window_loss = torch.zeros((), device=device)  # 当前梯度累积窗口内的平均 loss（见下）
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
            # loss 已除以 grad_accum，累加 grad_accum 个 micro-batch 后即为整个窗口的平均 loss。
            window_loss += loss.detach().float()

            if (micro_i + 1) % grad_accum == 0:
                lr_now = cosine_lr(opt_step, total_opt_steps, warmup_steps, base_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                cur = float(window_loss.item())  # 整个累积窗口的平均 loss
                last_micro = float(loss.item()) * grad_accum  # 旧口径：只取窗口内最后一个 micro-batch
                window_loss.zero_()
                pbar.update(1)
                pbar.set_postfix(loss=f"{cur:.4f}", lr=f"{lr_now:.2e}")
                if opt_step % tcfg["log_every"] == 0 or opt_step == total_opt_steps:
                    loss_curve.append(
                        {"step": opt_step, "loss": cur, "loss_last_micro": last_micro, "lr": lr_now}
                    )

    pbar.close()

    # ---- 合并 + 保存 ----
    ckpt_dir = abspath(cfg["paths"]["ckpt_dir"]) / args.method
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "qlora":
        # QLoRA: 不合并 4-bit base，仅保存 LoRA adapter 权重
        adapter_state: dict[str, torch.Tensor] = {}
        unwrap_qlora(model, adapter_state)
        adapter_dir = ckpt_dir / "lora_weights"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        torch.save(adapter_state, str(adapter_dir / "adapter_model.bin"))
    else:
        merge_adapter(model, args.method)
        save_dtype = torch_dtype(tcfg["save_merged_dtype"])
        model.to(save_dtype)
    if hasattr(model.config, "use_cache"):
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
            # final_loss = 最后 N 个记录点（每点是一个完整梯度累积窗口的平均）的均值，比单点稳健。
            "final_loss": _tail_mean(loss_curve, "loss", FINAL_LOSS_POINTS),
            "final_loss_window_points": min(len(loss_curve), FINAL_LOSS_POINTS),
            "final_loss_last_step": loss_curve[-1]["loss"] if loss_curve else None,
            # 旧口径（第一轮的 final_loss）：最后一个 micro-batch 的单点值，噪声大，仅供对照。
            "final_loss_last_micro": loss_curve[-1]["loss_last_micro"] if loss_curve else None,
            "wall_time_sec": time.time() - t0,
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 metrics：{out_path}")


class _nullctx:
    """在不需要自动混合精度时提供不抑制异常的空上下文。"""
    def __enter__(self):
        """进入空上下文，返回 None。"""
        return None

    def __exit__(self, *a):
        """返回 False，将上下文内异常继续向外传播。"""
        return False


if __name__ == "__main__":
    main()
