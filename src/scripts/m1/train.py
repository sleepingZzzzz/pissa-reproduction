"""[M1] 训练：用 我们的 PiSSA / LoRA / Full-FT 在 MetaMathQA 上微调 LLaMA-2-7B。

每种方法的精度/优化器/显存策略由 config 的 methods.<method> 决定。
训练后把适配器合并回权重并 save_pretrained，供 eval_gsm8k.py 评测。

训练目标与边界（HANDOFF T1/T2/T4/T5/T6/T7/P3/P4）：
    - 提示/回答按官方协议联合分词（data_ops.tokenize_example）；
    - 因果移位后无有效目标的样本在建 DataLoader 前过滤并计数；
    - 每个完整梯度累积窗口的目标 = 窗口内全部有效目标 token 的交叉熵之和 / 窗口有效 token 总数；
    - 不足一个完整窗口的尾部直接丢弃（不前向、不更新），每个 epoch 窗口状态独立；
    - 零次完整参数更新视为失败，不保存合并产物；
    - 学习率调度对齐 HF cosine：warmup 步数向上取整，第 k 步的学习率取 lambda(k)；
    - 非有限 loss/梯度立即报错停止，不静默继续更新。

用法：
    python src/scripts/m1/train.py --config configs/paper/m1_llama2_7b.yaml --method pissa
    python src/scripts/m1/train.py --config configs/paper/m1_llama2_7b.yaml --method lora --limit 2048 \\
        --set methods.lora.per_device_batch=2 --set methods.lora.grad_accum=4
    python src/scripts/m1/train.py --config configs/paper/m1_llama2_7b.yaml --method full
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接执行文件时，将 src 加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import json
import math
import subprocess
import time
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data_ops import (
    TRUNC_CATEGORIES,
    Collator,
    classify_truncation,
    read_jsonl,
    tokenize_example,
)
from utils import abspath, apply_overrides, load_config, set_seed, torch_dtype
from scripts.model_ops import (
    MASTER_WEIGHT_MODES,
    OPTIMIZER_KINDS,
    apply_adapter,
    build_optimizer,
    fix_unregistered_meta_tensors,
    load_base_model,
    load_tokenizer,
    merge_adapter,
    resolve_model_path,
)
from lora.inject_nf4 import unwrap_qlora

METHODS = ["pissa", "lora", "full", "qpissa", "qlora"]
FINAL_LOSS_POINTS = 10  # final_loss 取最后多少个记录点的均值
LOSS_NORMALIZATION = "window_valid_token_mean"


def hf_cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    """返回第 step 个优化步（从 0 起）使用的学习率，对齐 HF cosine + warmup。

    与 transformers.get_cosine_schedule_with_warmup（num_cycles=0.5）的 lambda 一致：
    step < warmup 时为 step / max(1, warmup)（第 0 步学习率为 0）；
    之后为 0.5 * (1 + cos(pi * progress))，progress = (step - warmup) / max(1, total - warmup)。

    Args:
        step: 从零开始的优化步编号。
        total: 总优化步数。
        warmup: warmup 步数（调用方按向上取整得到）。
        base_lr: 峰值学习率。
    """
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * 0.5 * progress)))


# 旧名称保留给既有调用方；行为已改为 HF 口径。
cosine_lr = hf_cosine_lr


def warmup_steps_for(total_opt_steps: int, warmup_ratio: float) -> int:
    """warmup 步数 = ceil(总步数 * warmup_ratio)，与 HF 标准一致。"""
    return int(math.ceil(total_opt_steps * warmup_ratio))


def resolve_sample_count(n_cfg, limit, n_available: int) -> tuple[int, int | None]:
    """T6：本次请求的样本数 = 配置值与 --limit 的较小者，再与可用数比较。

    Args:
        n_cfg: data.num_train_samples，None 表示不限制。
        limit: 命令行 --limit，None 表示不限制。
        n_available: 文件中的可用样本数。

    Returns:
        (实际使用数, 请求数)。请求数为 None 表示未设置上限；
        可用数少于请求数时仍按可用数返回，由调用方告警。

    Raises:
        ValueError: 请求数不是正整数。
    """
    requests = []
    for name, v in (("data.num_train_samples", n_cfg), ("--limit", limit)):
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} 必须是正整数，收到 {v!r}")
        requests.append(v)
    requested = min(requests) if requests else None
    used = n_available if requested is None else min(requested, n_available)
    return used, requested


def plan_updates(n_examples: int, per_device_batch: int, grad_accum: int, num_epochs: int) -> dict:
    """T1：按 drop_last 的 DataLoader 和完整窗口规则计算每 epoch 的窗口与丢弃数。

    Returns:
        micro_per_epoch、windows_per_epoch、total_opt_steps、
        dropped_samples_per_epoch（DataLoader 丢弃的不足一个 micro-batch 的样本）、
        dropped_micro_per_epoch（不足一个窗口的尾部 micro-batch 数）、
        dropped_tail_samples_per_epoch（尾部 micro-batch 对应的样本数）。
    """
    micro = n_examples // per_device_batch
    windows = micro // grad_accum
    return {
        "micro_per_epoch": micro,
        "windows_per_epoch": windows,
        "total_opt_steps": windows * num_epochs,
        "dropped_samples_per_epoch": n_examples - micro * per_device_batch,
        "dropped_micro_per_epoch": micro - windows * grad_accum,
        "dropped_tail_samples_per_epoch": (micro - windows * grad_accum) * per_device_batch,
    }


def iter_windows(loader, grad_accum: int):
    """逐个产出完整的 grad_accum 个 micro-batch 窗口；epoch 末尾不足的窗口被丢弃。"""
    buf: list = []
    for batch in loader:
        buf.append(batch)
        if len(buf) == grad_accum:
            yield buf
            buf = []


def valid_token_count(labels: torch.Tensor) -> int:
    """因果移位后的有效目标数：labels[:, 1:] 中不等于 -100 的元素个数。"""
    return int((labels[:, 1:] != -100).sum().item())


def validate_cfg(cfg: dict, method: str) -> None:
    """T7：训练前的配置有效性检查，不自动修正实验条件。

    Raises:
        ValueError: 字段缺失、类型/取值非法，或 lr_scheduler 不是已实现的 cosine。
    """
    mcfg = cfg["methods"][method]
    tcfg = cfg["train"]
    ocfg = cfg["optim"]
    if ocfg.get("lr_scheduler") != "cosine":
        raise ValueError(f"lr_scheduler 只支持已实现的 cosine，收到 {ocfg.get('lr_scheduler')!r}")
    for name, v in (
        ("methods.per_device_batch", mcfg["per_device_batch"]),
        ("methods.grad_accum", mcfg["grad_accum"]),
        ("train.num_epochs", tcfg["num_epochs"]),
        ("train.log_every", tcfg["log_every"]),
        ("train.max_seq_len", tcfg["max_seq_len"]),
    ):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} 必须是 >=1 的整数，收到 {v!r}")
    ratio = ocfg["warmup_ratio"]
    if not (0.0 <= float(ratio) < 1.0):
        raise ValueError(f"optim.warmup_ratio 必须在 [0, 1)，收到 {ratio!r}")
    if mcfg["optimizer"] not in OPTIMIZER_KINDS:
        raise ValueError(f"methods.{method}.optimizer 必须是 {OPTIMIZER_KINDS}，收到 {mcfg['optimizer']!r}")
    base_lr = mcfg.get("learning_rate") or ocfg["learning_rate"]
    if not (isinstance(base_lr, (int, float)) and base_lr > 0 and math.isfinite(base_lr)):
        raise ValueError(f"学习率必须是有限正数，收到 {base_lr!r}")


def resolve_autocast(mcfg: dict, dtype: torch.dtype):
    """返回 autocast 精度：methods.<m>.autocast_dtype 显式给出时优先（null 表示关闭），否则沿用 BF16/FP16 加载精度。"""
    if "autocast_dtype" in mcfg:
        v = mcfg["autocast_dtype"]
        return None if v in (None, "none") else torch_dtype(v)
    return dtype if dtype in (torch.bfloat16, torch.float16) else None


def code_version() -> dict:
    """返回当前 git HEAD 与工作区是否有未提交修改；不可用时给出 None。"""
    root = Path(__file__).resolve().parents[3]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
        return {"git_head": head, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"git_head": None, "dirty": None}


def env_info() -> dict:
    """记录解释器与关键依赖版本、设备信息。"""
    import importlib.metadata as md

    info = {"python": sys.version.split()[0], "torch": str(torch.__version__)}  # TorchVersion 是 str 子类，safe_dump 不认
    for pkg in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = None
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_runtime"] = str(torch.version.cuda)
    return info


def _tail_mean(curve: list[dict], key: str, n: int) -> float | None:
    """返回曲线末 n 个记录中 key 字段的算术平均；曲线为空时返回 None。"""
    tail = curve[-n:]
    return sum(c[key] for c in tail) / len(tail) if tail else None


def build_examples(cfg: dict, tokenizer, limit: int | None) -> dict:
    """读取训练集并分词，统计截断分类与长度分布，过滤零监督样本（T4）。

    与训练、诊断脚本共用同一实现，保证它们看到的样本完全一致。

    Args:
        cfg: 含 data/train 配置的字典。
        tokenizer: 已加载的分词器。
        limit: 命令行样本上限，与 data.num_train_samples 取较小者；None 不限。

    Returns:
        字典：examples（过滤后）、trunc_counts、zero_supervision、n_used、n_requested、
        n_available、length_stats（未截断总长度的均值/分位数/最大值与超长计数）。
    """
    tcfg = cfg["train"]
    rows = read_jsonl(abspath(cfg["data"]["train_jsonl"]))
    n_cfg = cfg["data"].get("num_train_samples")
    n_used, n_requested = resolve_sample_count(n_cfg, limit, len(rows))
    if n_requested is not None and len(rows) < n_requested:
        print(f"  [warn] jsonl 只有 {len(rows)} 条，少于请求的 {n_requested} 条（num_train_samples/--limit 较小者）")
    n_available = len(rows)
    rows = rows[:n_used]
    max_len = tcfg["max_seq_len"]
    resp_only = tcfg["loss_on_response_only"]

    examples: list[dict] = []
    trunc_counts = {c: 0 for c in TRUNC_CATEGORIES}
    zero_supervision = 0
    full_lens: list[int] = []
    for r in tqdm(rows, desc="分词", unit="ex"):
        info = classify_truncation(tokenizer, r["instruction"], r["response"], max_len)
        trunc_counts[info["category"]] += 1
        full_lens.append(info["full_len"])
        ex = tokenize_example(tokenizer, r["instruction"], r["response"], max_len, resp_only)
        if info["valid_targets"] == 0 and resp_only:
            zero_supervision += 1  # T4：零有效目标样本在建 DataLoader 前过滤
            continue
        examples.append(ex)

    srt = sorted(full_lens)

    def pct(q: float):
        return srt[min(len(srt) - 1, int(q * len(srt)))] if srt else None

    length_stats = {
        "mean": sum(srt) / len(srt) if srt else None,
        "p50": pct(0.50), "p90": pct(0.90), "p95": pct(0.95), "p99": pct(0.99),
        "max": srt[-1] if srt else None,
        "num_over_max_len": sum(1 for v in srt if v > max_len),
    }
    return {
        "examples": examples, "trunc_counts": trunc_counts, "zero_supervision": zero_supervision,
        "n_used": n_used, "n_requested": n_requested, "n_available": n_available,
        "length_stats": length_stats,
    }


def main() -> None:
    """执行配置驱动的微调，合并或导出适配器，保存 checkpoint、配置快照与训练指标。

    先完成数据处理与更新数量检查（零更新直接失败），再加载模型训练。
    调用会加载模型、可能下载权重并训练；导入模块不会自动执行。
    """
    parser = argparse.ArgumentParser(description="[M1] 训练 PiSSA/LoRA/Full")
    parser.add_argument("--config", default="configs/paper/m1_llama2_7b.yaml")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--limit", type=int, default=None, help="样本上限，与 data.num_train_samples 取较小者")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖配置字段，如 methods.pissa.dtype=float32；可重复，生效值写入快照")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的 checkpoint/指标")
    parser.add_argument("--master-weights", default="none", choices=list(MASTER_WEIGHT_MODES),
                        help="none=优化器直接更新参数；cpu_fp32=BF16 参数 + CPU 上的 FP32 主权重与 Adam 状态"
                             "（复现官方 DeepSpeed bf16 混合精度，仅 full 方法）")
    parser.add_argument("--tokenize-only", action="store_true",
                        help="只做分词与截断统计并保存 JSON，不加载模型、不训练；不会读写 checkpoint")
    args = parser.parse_args()

    cfg = load_config(abspath(args.config))
    apply_overrides(cfg, args.set)
    mcfg = cfg["methods"][args.method]
    tcfg = cfg["train"]
    ocfg = cfg["optim"]
    validate_cfg(cfg, args.method)
    if args.master_weights != "none":
        if args.method != "full" or mcfg["dtype"] != "bfloat16":
            raise SystemExit("--master-weights cpu_fp32 只用于 methods.full.dtype=bfloat16 的 full 方法。")
        mcfg["master_weights"] = args.master_weights  # 写入有效配置，随快照保存

    ckpt_dir = abspath(cfg["paths"]["ckpt_dir"]) / args.method
    metrics_dir = abspath(cfg["paths"]["output_dir"]) / "metrics"
    out_path = metrics_dir / f"train_{args.method}.json"
    if not args.tokenize_only and not args.overwrite and (
            out_path.exists() or (ckpt_dir / "config.json").exists() or (ckpt_dir / "lora_weights").exists()):
        raise SystemExit(f"已存在 {out_path} 或 {ckpt_dir} 中的产物；请更换输出路径或显式传 --overwrite。")

    set_seed(tcfg["seed"])
    t_start = time.time()

    # TF32 由配置控制（默认保持开启以兼容既有运行）；官方 FP32 未开启 TF32，差异记录在 P10。
    use_tf32 = bool(tcfg.get("tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch_dtype(mcfg["dtype"])
    amp_dtype = resolve_autocast(mcfg, dtype)
    print(f"方法={args.method} 设备={device} dtype={mcfg['dtype']} autocast={amp_dtype} tf32={use_tf32}")

    # ---- 分词器与数据（先于模型加载，便于零更新时尽早失败）----
    model_path = resolve_model_path(cfg)
    trc = cfg.get("model", {}).get("trust_remote_code", False)
    low_cpu = cfg.get("model", {}).get("low_cpu_mem_usage", True)
    tokenizer = load_tokenizer(model_path, trust_remote_code=trc)

    built = build_examples(cfg, tokenizer, args.limit)
    examples, trunc_counts = built["examples"], built["trunc_counts"]
    zero_supervision, n_used, n_requested = built["zero_supervision"], built["n_used"], built["n_requested"]
    max_len = tcfg["max_seq_len"]

    per_device = mcfg["per_device_batch"]
    grad_accum = mcfg["grad_accum"]
    plan = plan_updates(len(examples), per_device, grad_accum, tcfg["num_epochs"])
    total_opt_steps = plan["total_opt_steps"]
    if total_opt_steps < 1 and not args.tokenize_only:
        raise SystemExit(
            f"过滤后样本 {len(examples)} 条不足一个完整更新窗口 "
            f"(per_device_batch*grad_accum={per_device * grad_accum})：零次参数更新，不作为正常训练。"
        )
    warmup_steps = warmup_steps_for(total_opt_steps, ocfg["warmup_ratio"])
    base_lr = mcfg.get("learning_rate") or ocfg["learning_rate"]
    window_samples = per_device * grad_accum
    print(
        f"  原始选中={n_used} 零监督剔除={zero_supervision} 过滤后={len(examples)}  "
        f"micro_bs={per_device} grad_accum={grad_accum} → 有效batch={window_samples}; "
        f"每epoch完整窗口={plan['windows_per_epoch']} 总更新步={total_opt_steps} warmup={warmup_steps}"
    )
    print(
        f"  每epoch丢弃：DataLoader 样本={plan['dropped_samples_per_epoch']}，"
        f"尾部窗口 micro-batch={plan['dropped_micro_per_epoch']}（样本 {plan['dropped_tail_samples_per_epoch']}）"
    )
    print(f"  截断分类（max_len={max_len}）：{trunc_counts}")
    print(f"  未截断总长度统计：{built['length_stats']}")
    if args.tokenize_only:
        stats_path = metrics_dir / f"tokenize_stats_len{max_len}_n{n_used}.json"
        if stats_path.exists() and not args.overwrite:
            raise SystemExit(f"已存在 {stats_path}；请显式传 --overwrite。")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps({
            "experiment": "M1/tokenize_stats",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": {"config_path": str(abspath(args.config)), "overrides": args.set, "max_seq_len": max_len,
                       "loss_on_response_only": tcfg["loss_on_response_only"],
                       "num_samples_requested": n_requested, "num_samples_selected": n_used,
                       "tokenizer_path": model_path, "code": code_version()},
            "results": {
                "num_zero_supervision_removed": zero_supervision,
                "num_after_filter": len(examples),
                "truncation_counts": trunc_counts,
                "truncation_ratio": {k: v / max(n_used, 1) for k, v in trunc_counts.items()},
                "length_stats": built["length_stats"],
                "plan_if_trained": plan,
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存分词统计：{stats_path}（未加载模型）")
        return

    # ---- 模型 ----
    # QLoRA：按 4-bit 加载（非量化模块保持 methods.qlora.dtype，论文为 FP32，与官方 prepare_model_for_kbit_training
    # 把 fp16/bf16 的非量化参数升到 FP32 的结果一致）；QPiSSA：先按 methods.qpissa.dtype 浮点加载，
    # 由 apply_adapter 逐层做交替初始化并重新量化残差。
    is_quant = args.method in ("qlora", "qpissa")
    if is_quant and device != "cuda":
        raise SystemExit(f"{args.method} 需要 CUDA（bitsandbytes 4-bit）。")
    if args.method == "qlora":
        from transformers import BitsAndBytesConfig

        q = cfg["quantization"]
        if not q.get("load_in_4bit"):
            raise SystemExit("qlora 要求 quantization.load_in_4bit=true。")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype(q.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        )
        model = load_base_model(model_path, dtype, trust_remote_code=trc, low_cpu_mem_usage=False,
                                quantization_config=quant_cfg, device_map={"": torch.cuda.current_device()})
    else:
        model = load_base_model(model_path, dtype, trust_remote_code=trc, low_cpu_mem_usage=low_cpu)
        model.to(device)
    # Baichuan2 等模型的 RotaryEmbedding 缓存是普通 tensor 属性，model.to() 不会移动。
    fix_unregistered_meta_tensors(model, device)
    if mcfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        if args.method != "full":
            # 默认 use_reentrant=True 时，冻结的 embedding 输出不需要梯度，检查点内的适配器会得不到梯度
            # （peft 的 prepare_model_for_kbit_training 同样调用该方法）。
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    stats = apply_adapter(model, args.method, cfg)
    print(
        f"  替换层数={stats['replaced_modules']}  可训练参数={stats['trainable_params']:,}"
        f"  占比={stats['trainable_pct']:.4f}%"
    )

    param_dtype_numel: dict[str, int] = {}
    for p in model.parameters():
        key = f"{'nf4_packed_' if p.dtype == torch.uint8 else ''}{str(p.dtype).replace('torch.', '')}"
        param_dtype_numel[key] = param_dtype_numel.get(key, 0) + p.numel()
    print(f"  参数精度分布（元素数）：{param_dtype_numel}")

    # DataLoader 使用独立的数据顺序随机源，不受适配器初始化消耗全局 RNG 的影响。
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(int(tcfg["seed"]))
    loader = DataLoader(
        examples,
        batch_size=per_device,
        shuffle=True,
        generator=shuffle_gen,
        collate_fn=Collator(tokenizer.pad_token_id),
        drop_last=True,
    )
    assert len(loader) == plan["micro_per_epoch"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer, opt_used = build_optimizer(trainable, base_lr, ocfg["weight_decay"], mcfg["optimizer"],
                                      master_weights=args.master_weights)

    # ---- 训练循环 ----
    model.train()
    loss_curve: list[dict] = []
    opt_step = 0
    total_valid_tokens = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_opt_steps, desc=f"训练[{args.method}]", unit="step")
    for epoch in range(tcfg["num_epochs"]):
        for window in iter_windows(loader, grad_accum):
            counts = [valid_token_count(b["labels"]) for b in window]
            window_tokens = sum(counts)
            if window_tokens == 0:
                raise RuntimeError(f"epoch {epoch} 更新步 {opt_step}：窗口内没有任何有效目标 token（过滤应已排除）。")
            window_loss = torch.zeros((), device=device)
            for batch, n_tok in zip(window, counts):
                if n_tok == 0:
                    continue  # 该 micro-batch 无贡献，跳过前向避免 0/0
                batch = {k: v.to(device) for k, v in batch.items()}
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype is not None and device == "cuda") else _nullctx()
                with ctx:
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                micro_mean = out.loss.float()  # 本 micro-batch 有效 token 的平均 CE
                if not torch.isfinite(micro_mean):
                    raise RuntimeError(f"非有限 loss：epoch {epoch} 更新步 {opt_step}，值={micro_mean.item()}")
                weight = n_tok / window_tokens  # sum_CE_micro / window_tokens = micro_mean * n_tok / window_tokens
                (micro_mean * weight).backward()
                window_loss += micro_mean.detach() * weight

            lr_now = hf_cosine_lr(opt_step, total_opt_steps, warmup_steps, base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            if opt_step == 0:
                no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                if no_grad:
                    raise RuntimeError(f"首个窗口后 {len(no_grad)} 个可训练参数没有梯度（如 {no_grad[:3]}），"
                                       "检查点/冻结配置有误，终止训练。")
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"非有限梯度范数：epoch {epoch} 更新步 {opt_step}，值={float(grad_norm)}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            total_valid_tokens += window_tokens
            cur = float(window_loss.item())
            pbar.update(1)
            pbar.set_postfix(loss=f"{cur:.4f}", lr=f"{lr_now:.2e}")
            if opt_step % tcfg["log_every"] == 0 or opt_step == total_opt_steps:
                loss_curve.append({"step": opt_step, "loss": cur, "lr": lr_now,
                                   "grad_norm": float(grad_norm), "window_tokens": window_tokens})
    pbar.close()
    if opt_step != total_opt_steps:
        raise RuntimeError(f"实际更新步数 {opt_step} 与计划 {total_opt_steps} 不一致，不保存产物。")
    t_train = time.time() - t0

    # ---- 合并 + 保存 ----
    # 保存前释放优化器状态：cpu_fp32 主权重占用数十 GiB 锁页内存，与保存时的主机拷贝叠加会触发 OOM。
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 锁页内存由 PyTorch 的 CachingHostAllocator 缓存，del 后不会归还系统，需显式清空。
    if hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
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

    # ---- 配置快照（随 checkpoint 保存，评测可据此恢复训练时配置）----
    snapshot = {
        "method": args.method,
        "argv": sys.argv,
        "config_path": str(abspath(args.config)),
        "overrides": args.set,
        "effective_config": cfg,
        "code": code_version(),
        "env": env_info(),
    }
    (ckpt_dir / "run_config.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # ---- metrics ----
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": f"M1/train ({args.method})",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "method": args.method,
            "dtype": mcfg["dtype"],
            "autocast_dtype": str(amp_dtype),
            "param_dtype_numel": param_dtype_numel,
            "tf32": use_tf32,
            "optimizer": opt_used,
            "master_weights": args.master_weights,
            "grad_checkpointing": mcfg["grad_checkpointing"],
            "per_device_batch": per_device,
            "grad_accum": grad_accum,
            "effective_batch": window_samples,
            "lr": base_lr,
            "lr_schedule": "hf_cosine_ceil_warmup",
            "warmup_steps": warmup_steps,
            "loss_normalization": LOSS_NORMALIZATION,
            "num_samples_requested": n_requested,
            "num_samples_selected": n_used,
            "num_zero_supervision_removed": zero_supervision,
            "num_samples": len(examples),
            "total_opt_steps": total_opt_steps,
            "actual_opt_steps": opt_step,
            "samples_used_for_updates": opt_step * window_samples,
            "dropped_per_epoch": {k: plan[k] for k in (
                "dropped_samples_per_epoch", "dropped_micro_per_epoch", "dropped_tail_samples_per_epoch")},
            "total_valid_target_tokens": total_valid_tokens,
            "truncation_counts": trunc_counts,
            "max_seq_len": max_len,
            "r": cfg["adapter"]["r"],
            "seed": tcfg["seed"],
            "code": code_version(),
        },
        "param_stats": stats,
        "results": {
            "loss_curve": loss_curve,
            # final_loss = 最后 N 个记录点（每点是一个完整梯度累积窗口的 token 加权平均）的均值，不是整轮平均。
            "final_loss": _tail_mean(loss_curve, "loss", FINAL_LOSS_POINTS),
            "final_loss_window_points": min(len(loss_curve), FINAL_LOSS_POINTS),
            "final_loss_last_step": loss_curve[-1]["loss"] if loss_curve else None,
            "train_loop_sec": t_train,
            # wall_time_sec 含训练循环与合并保存，不含模型/数据准备。
            "wall_time_sec": time.time() - t0,
            "total_sec_including_prep": time.time() - t_start,
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
