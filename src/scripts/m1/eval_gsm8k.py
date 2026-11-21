"""[M1] 在 GSM8K test 上评测训练好的合并模型。

评测协议对齐官方评测参考 f7509eee（HANDOFF P8）：
    - 提示为 PROMPT_TEMPLATE（以 ``### Response:`` 结尾，无换行），不截断；
    - 贪心、单回答，生成预算与停止串由配置 eval.max_new_tokens / eval.stop_strings 决定；
    - 主评分为官方提取/判定（gsm8k_score.score_primary），副评分为 decimal_v1；
    - 无效 gold（无 ``####`` 标记等）在生成前统一排除并记数，pred=None 仍计为未答对；
    - 逐题生成档案（jsonl）与指标 JSON 分开保存，可无需重新生成而离线重评分。

后端：
    --backend hf    transformers.generate（默认；QLoRA 唯一可用后端）
    --backend vllm  vLLM（需在装有 vllm 的环境中运行，例如 /root/autodl-tmp/venv-vllm/bin/python）

用法：
    python src/scripts/m1/eval_gsm8k.py --method pissa --ckpt checkpoints/m1/pissa
    python src/scripts/m1/eval_gsm8k.py --config configs/paper/m1_llama2_7b.yaml --method lora --limit 100
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接执行文件时，将 src 加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

from scripts.data_ops import PROMPT_TEMPLATE, read_jsonl
from scripts.m1 import gsm8k_score as gs
from utils import abspath, apply_overrides, load_config, torch_dtype
from scripts.model_ops import fix_unregistered_meta_tensors, load_tokenizer, resolve_model_path
from lora.inject_nf4 import inject_lora_4bit as inject_qlora_eval

METRIC_DEFINITIONS = {
    "N_raw": "原始评测题数（计数）。",
    "N_gold": "标准答案可解析（含 #### 且为有限数）的题数；无效 gold 在生成前排除。",
    "N_pred": "有效 gold 题中，主评分能提取到预测数值的题数。",
    "N_correct": "有效 gold 题中，预测与 gold 数值相等的题数。",
    "accuracy": "主准确率 = N_correct / N_gold。pred=None 仍在分母中，计为未答对。",
    "gold_valid_ratio": "N_gold / N_raw。",
    "pred_parse_coverage": "N_pred / N_gold。",
    "accuracy_on_parsed": "N_correct / N_pred；N_pred=0 时为 null（未定义），不以 0 代替。",
    "teaching_example（非本次数据）": "100 题 → 90 有效 gold → 72 可解析 pred → 54 正确："
                                      "有效比例 90%，主准确率 60%，覆盖率 80%，可解析子集准确率 75%。",
}


def resolve_stop_strings(spec) -> list[str]:
    """eval.stop_strings：缺省或 'official_f7509eee' 取官方评测参考的列表；'none'/[] 为不设停止串。"""
    if spec is None or spec == "official_f7509eee":
        return list(gs.OFFICIAL_STOP_STRINGS_F7509EEE)
    if spec in ("none", []):
        return []
    if isinstance(spec, list) and all(isinstance(s, str) for s in spec):
        return list(spec)
    raise ValueError(f"无法解析 eval.stop_strings={spec!r}")


def compute_metrics(n_raw: int, n_gold: int, n_pred: int, n_correct: int) -> dict:
    """按 E3 的定义由四个计数计算指标；分母为零时给出 None 而非 0。"""
    return {
        "N_raw": n_raw,
        "N_gold": n_gold,
        "N_pred": n_pred,
        "N_correct": n_correct,
        "accuracy": (n_correct / n_gold) if n_gold else None,
        "gold_valid_ratio": (n_gold / n_raw) if n_raw else None,
        "pred_parse_coverage": (n_pred / n_gold) if n_gold else None,
        "accuracy_on_parsed": (n_correct / n_pred) if n_pred else None,
    }


def check_pairing(n_prompts: int, outputs: list) -> None:
    """E4：输出条数必须与输入题数一致，保证一题一答；不允许 zip 静默截短。"""
    if len(outputs) != n_prompts:
        raise RuntimeError(f"生成返回 {len(outputs)} 条，但输入 {n_prompts} 题；一题一答契约被破坏。")


def validate_qlora_restore(model, replaced: list[str], adapter_sd: dict) -> None:
    """M3-1：校验 QLoRA 适配器恢复——键集合、shape 与实际加载结果。

    Raises:
        RuntimeError: 没有任何目标被注入、适配器文件键集合与预期不符、shape 不符，
            或 load_state_dict 返回了意外键/缺失了适配器键。
    """
    if not replaced:
        raise RuntimeError("QLoRA 评测：没有任何目标层被注入。")
    expected = {f"{n}.lora_{ab}" for n in replaced for ab in ("A", "B")}
    got = set(adapter_sd)
    if got != expected:
        raise RuntimeError(
            f"适配器文件键集合与预期不符：缺少 {sorted(expected - got)[:3]}…，多余 {sorted(got - expected)[:3]}…"
        )
    params = dict(model.named_parameters())
    for k, v in adapter_sd.items():
        if tuple(params[k].shape) != tuple(v.shape):
            raise RuntimeError(f"{k} shape 不符：模型 {tuple(params[k].shape)} vs 文件 {tuple(v.shape)}")
    result = model.load_state_dict(adapter_sd, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"load_state_dict 出现意外键：{list(result.unexpected_keys)[:3]}")
    missing_adapter = [k for k in result.missing_keys if "lora_" in k]
    if missing_adapter:
        raise RuntimeError(f"适配器键未被加载：{missing_adapter[:3]}")


def generate_hf(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int,
                stops: list[str], device: str, desc: str):
    """HF 贪心生成；返回每题 (文本, 新增 token 数, 是否触达预算, 是否生成 EOS, 命中的停止串)。"""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    records = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            # 必须显式传 attention_mask：pad_token 与 eos 可能同 ID，无法从 input_ids 推断。
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            pad_token_id=pad_id,
        )
        if stops:
            gen_kwargs.update(stop_strings=stops, tokenizer=tokenizer)
        with torch.no_grad():
            gen = model.generate(**gen_kwargs)
        if gen.dim() != 2 or gen.shape[0] != len(chunk):
            raise RuntimeError(f"生成返回形状 {tuple(gen.shape)}，期望 ({len(chunk)}, L)")
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        for row in new_tokens:
            ids = row.tolist()
            if eos_id in ids:
                n_new, has_eos = ids.index(eos_id) + 1, True
            else:
                has_eos = False
                n_new = len(ids) if pad_id == eos_id else sum(1 for t in ids if t != pad_id)
            text = tokenizer.decode(ids[:n_new], skip_special_tokens=True)
            text, hit = gs.apply_stop(text, stops)
            records.append({
                "output": text,
                "new_tokens": n_new,
                "hit_budget": n_new >= max_new_tokens,
                "has_eos": has_eos,
                "stop_string": hit,
            })
    check_pairing(len(prompts), records)
    return records


def generate_vllm(ckpt_dir: str, prompts: list[str], max_tokens: int, stops: list[str], dtype: str,
                  tokenizer_path: str):
    """vLLM 贪心生成（temperature=0, top_p=1, n=1）；停止串不保留在输出中。

    tokenizer_path 使用原始基座模型的分词器：checkpoint 由 transformers 5.x 保存，其
    tokenizer_config 的 tokenizer_class=TokenizersBackend 不被 vLLM 环境的 transformers 4.x 识别。
    """
    import os

    # 父进程已初始化 CUDA 时，默认 fork 会报 "Cannot re-initialize CUDA in forked subprocess"。
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    llm = LLM(model=ckpt_dir, tokenizer=tokenizer_path, dtype=dtype, tensor_parallel_size=1)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, stop=stops or None, n=1)
    completions = llm.generate(prompts, params)
    check_pairing(len(prompts), completions)
    records = []
    for c in completions:
        if len(c.outputs) != 1:
            raise RuntimeError(f"一题返回 {len(c.outputs)} 个回答，期望 1。")
        o = c.outputs[0]
        records.append({
            "output": o.text,
            "new_tokens": len(o.token_ids),
            "hit_budget": o.finish_reason == "length",
            "has_eos": o.finish_reason == "stop" and o.stop_reason is None,
            "stop_string": o.stop_reason if isinstance(o.stop_reason, str) else None,
        })
    return records


def main() -> None:
    """加载指定方法 checkpoint，生成 GSM8K 回答，按官方规则评分并保存档案与指标。

    命令行可覆盖 checkpoint 或限制评测条数。不根据测试集选择超参数、规则或 checkpoint。
    """
    parser = argparse.ArgumentParser(description="[M1] GSM8K 评测")
    parser.add_argument("--config", default=None, help="缺省时读取 checkpoint 随附的 run_config.yaml")
    parser.add_argument("--method", required=True, choices=["pissa", "lora", "full", "qpissa", "qlora"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ckpt", default=None, help="覆盖默认 checkpoint 目录")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有评测产物")
    parser.add_argument("--fallback-score", action="store_true",
                        help="额外计算回退评分 last_number_v1（取全文最后一个数字），写入 fallback_secondary；"
                             "辅助指标，不替代主评分；默认关闭，关闭时输出与以前完全一致")
    args = parser.parse_args()

    # ---- 配置：显式 --config，或 checkpoint 快照 ----
    if args.config is not None:
        cfg = load_config(abspath(args.config))
        ckpt_dir = Path(args.ckpt) if args.ckpt else abspath(cfg["paths"]["ckpt_dir"]) / args.method
        config_source = f"--config {args.config}"
    else:
        if args.ckpt is None:
            raise SystemExit("未给 --config 时必须提供 --ckpt 以读取其 run_config.yaml。")
        ckpt_dir = Path(args.ckpt)
        snap_path = ckpt_dir / "run_config.yaml"
        if not snap_path.exists():
            raise SystemExit(f"{snap_path} 不存在；旧 checkpoint 请显式传 --config（训练配置未核验）。")
        cfg = yaml.safe_load(snap_path.read_text(encoding="utf-8"))["effective_config"]
        config_source = f"checkpoint snapshot {snap_path}"
    apply_overrides(cfg, args.set)
    snap = ckpt_dir / "run_config.yaml"
    if args.config is not None and snap.exists():
        trained = yaml.safe_load(snap.read_text(encoding="utf-8"))["effective_config"]
        for key in ("adapter", "quantization"):
            if trained.get(key) != cfg.get(key):
                msg = f"评测配置的 {key} 与 checkpoint 训练快照不一致：{cfg.get(key)} vs {trained.get(key)}"
                if args.method in ("qlora", "qpissa"):
                    raise SystemExit(msg)
                print(f"  [warn] {msg}")

    ecfg = cfg["eval"]
    max_new = int(ecfg["max_new_tokens"])
    stops = resolve_stop_strings(ecfg.get("stop_strings"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_dir = abspath(cfg["paths"]["output_dir"]) / "metrics"
    out_path = metrics_dir / f"eval_{args.method}.json"
    gen_path = metrics_dir / f"eval_{args.method}_generations.jsonl"
    if not args.overwrite and (out_path.exists() or gen_path.exists()):
        raise SystemExit(f"已存在 {out_path} 或 {gen_path}；请显式传 --overwrite。")
    if args.backend == "vllm" and args.method == "qlora":
        raise SystemExit("QLoRA 无合并检查点，vLLM 后端不可用；请用 --backend hf。")

    # ---- 题目与 gold 预检 ----
    rows = read_jsonl(abspath(cfg["data"]["gsm8k_jsonl"]))
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit 必须是正整数。")
        rows = rows[: args.limit]
    n_raw = len(rows)
    if n_raw == 0:
        raise SystemExit("评测集为空，无法完成有效评测。")
    items = []
    invalid_gold = []
    for i, r in enumerate(rows):
        g = gs.gold_value(r["answer"])
        if g is None:
            invalid_gold.append({"idx": i, "question": r["question"], "answer": r["answer"]})
        else:
            items.append({"idx": i, "question": r["question"], "answer": r["answer"], "gold": g})
    if not items:
        raise SystemExit("没有任何有效 gold，无法完成有效评测。")
    prompts = [PROMPT_TEMPLATE.format(instruction=it["question"]) for it in items]

    print(f"加载模型：{ckpt_dir}（backend={args.backend}，配置来源 {config_source}）")
    trc = cfg.get("model", {}).get("trust_remote_code", False)
    t0 = time.time()
    if args.backend == "vllm":
        records = generate_vllm(str(ckpt_dir), prompts, max_new, stops, cfg["train"]["save_merged_dtype"],
                                  resolve_model_path(cfg))
    else:
        from transformers import AutoModelForCausalLM

        tokenizer = load_tokenizer(str(ckpt_dir), trust_remote_code=trc)
        tokenizer.padding_side = "left"  # 生成时左填充
        if args.method == "qlora":
            acfg = cfg["adapter"]
            qcfg = cfg.get("quantization", {})
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=qcfg.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
                bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
            )
            if device != "cuda":
                raise SystemExit("QLoRA 评测需要 CUDA。")
            mcfg = cfg["methods"]["qlora"]
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_dir), quantization_config=bnb_config, dtype=torch_dtype(mcfg["dtype"]),
                low_cpu_mem_usage=False, trust_remote_code=trc, device_map={"": torch.cuda.current_device()},
            )
            fix_unregistered_meta_tensors(model, torch.device(device))
            replaced = inject_qlora_eval(
                model, r=acfg["r"], alpha=acfg["alpha"], target_modules=acfg["target_modules"],
                dtype=torch_dtype(mcfg.get("adapter_dtype", "float32")),
            )
            sd = torch.load(f"{ckpt_dir}/lora_weights/adapter_model.bin", map_location="cpu", weights_only=True)
            validate_qlora_restore(model, replaced, sd)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_dir), dtype=torch_dtype(cfg["train"]["save_merged_dtype"]),
                low_cpu_mem_usage=False, trust_remote_code=trc
            ).to(device)
            fix_unregistered_meta_tensors(model, device)
        model.eval()
        records = generate_hf(model, tokenizer, prompts, int(ecfg["batch_size"]), max_new, stops, device,
                              f"评测[{args.method}]")
    gen_sec = time.time() - t0

    # ---- 评分与归档（逐题档案与指标分开）----
    metrics_dir.mkdir(parents=True, exist_ok=True)
    n_pred = n_correct = n_correct_sec = n_correct_fb = n_budget = n_eos = 0
    with open(gen_path, "w", encoding="utf-8") as f:
        for it, prompt, rec in zip(items, prompts, records):
            ok, pred = gs.score_primary(rec["output"], it["gold"])
            ok2, cand = gs.score_secondary(rec["output"], it["answer"])
            n_pred += int(pred is not None)
            n_correct += int(ok)
            n_correct_sec += int(ok2)
            n_budget += int(rec["hit_budget"])
            n_eos += int(rec["has_eos"])
            row = {
                "idx": it["idx"], "question": it["question"], "prompt": prompt, "answer": it["answer"],
                "gold": it["gold"], "output": rec["output"], "pred": pred, "correct": ok,
                "decimal_candidate": cand, "decimal_correct": ok2,
                "new_tokens": rec["new_tokens"], "hit_budget": rec["hit_budget"],
                "has_eos": rec["has_eos"], "stop_string": rec["stop_string"],
            }
            if args.fallback_score:
                ok3, fb_cand = gs.score_fallback(rec["output"], it["gold"])
                n_correct_fb += int(ok3)
                row["fallback_candidate"], row["fallback_correct"] = fb_cand, ok3
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_gold = len(items)
    primary = compute_metrics(n_raw, n_gold, n_pred, n_correct)
    print(f"GSM8K 主准确率（{args.method}，{gs.SCORER_PRIMARY}）= {primary['accuracy']:.4f}  ({n_correct}/{n_gold}); "
          f"副评分 {gs.SCORER_SECONDARY} = {n_correct_sec / n_gold:.4f}")
    payload = {
        "experiment": f"M1/eval_gsm8k ({args.method})",
        "status": "complete",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "method": args.method,
            "backend": args.backend,
            "ckpt": str(ckpt_dir),
            "config_source": config_source,
            "num_eval": n_raw,
            "max_new_tokens": max_new,
            "stop_strings": stops,
            "prompt_template_tail": "### Response:",
            "decoding": "greedy, n=1",
            "scorer_primary": gs.SCORER_PRIMARY,
            "scorer_secondary": gs.SCORER_SECONDARY,
            "generations_file": str(gen_path),
            **({"scorer_fallback": gs.SCORER_FALLBACK} if args.fallback_score else {}),
        },
        "results": {
            "accuracy": primary["accuracy"],
            "correct": n_correct,
            "total": n_gold,
            **primary,
            "decimal_secondary": {"scorer": gs.SCORER_SECONDARY, "N_correct": n_correct_sec,
                                  "accuracy": n_correct_sec / n_gold},
            **({"fallback_secondary": {"scorer": gs.SCORER_FALLBACK, "N_correct": n_correct_fb,
                                       "accuracy": n_correct_fb / n_gold}} if args.fallback_score else {}),
            "invalid_gold_count": len(invalid_gold),
            "invalid_gold_examples": invalid_gold[:50],
            "hit_budget_count": n_budget,
            "has_eos_count": n_eos,
            "generation_sec": gen_sec,
        },
        "metric_definitions": METRIC_DEFINITIONS,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 metrics：{out_path}；生成档案：{gen_path}")


if __name__ == "__main__":
    main()
