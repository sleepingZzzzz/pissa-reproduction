"""[M1] 在 GSM8K test 上评测训练好的合并模型，计算 exact-match 准确率。

用法：
    python scripts/m1/eval_gsm8k.py --config configs/m1_llama2_7b.yaml --method pissa
    python scripts/m1/eval_gsm8k.py --config configs/m1_llama2_7b.yaml --method lora --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

import torch
from tqdm import tqdm

from common import (
    PROMPT_TEMPLATE,
    abspath,
    fix_unregistered_meta_tensors,
    load_config,
    load_tokenizer,
    read_jsonl,
    torch_dtype,
    inject_lora_4bit as inject_qlora_eval,
)

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """从文本中抽取最终数字答案：优先 'The answer is X'，否则取最后一个数字。"""
    m = re.search(r"answer is\s*\$?(-?\d[\d,]*\.?\d*)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _NUM.findall(text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


def gold_answer(answer_field: str) -> str | None:
    """GSM8K 金标答案在 '#### N' 之后。"""
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    return extract_answer(answer_field)


def main() -> None:
    parser = argparse.ArgumentParser(description="[M1] GSM8K 评测")
    parser.add_argument("--config", default="configs/m1_llama2_7b.yaml")
    parser.add_argument("--method", required=True, choices=["pissa", "lora", "full", "qpissa", "qlora"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ckpt", default=None, help="覆盖默认 checkpoint 目录")
    args = parser.parse_args()

    cfg = load_config(abspath(args.config))
    ecfg = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = args.ckpt or str(abspath(cfg["paths"]["ckpt_dir"]) / args.method)
    print(f"加载模型：{ckpt_dir}")
    trc = cfg.get("model", {}).get("trust_remote_code", False)
    tokenizer = load_tokenizer(ckpt_dir, trust_remote_code=trc)
    tokenizer.padding_side = "left"  # 生成时左填充
    from transformers import AutoModelForCausalLM

    if args.method == "qlora":
        # QLoRA eval: 加载 4-bit base model + 注入 LoRA + 加载 adapter weights
        acfg = cfg["adapter"]
        qcfg = cfg.get("quantization", {})
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qcfg.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, quantization_config=bnb_config, low_cpu_mem_usage=True, trust_remote_code=trc
        )
        fix_unregistered_meta_tensors(model, torch.device("cuda"))
        inject_qlora_eval(model, r=acfg["r"], alpha=acfg["alpha"], target_modules=acfg["target_modules"])
        sd = torch.load(f"{ckpt_dir}/lora_weights/adapter_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, dtype=torch_dtype(cfg["train"]["save_merged_dtype"]),
            low_cpu_mem_usage=False, trust_remote_code=trc
        ).to(device)
        fix_unregistered_meta_tensors(model, device)
    model.eval()

    rows = read_jsonl(abspath(cfg["data"]["gsm8k_jsonl"]))
    if args.limit is not None:
        rows = rows[: args.limit]

    bs = ecfg["batch_size"]
    correct = 0
    samples: list[dict] = []
    pbar = tqdm(range(0, len(rows), bs), desc=f"评测[{args.method}]", unit="batch")
    for start in pbar:
        chunk = rows[start : start + bs]
        prompts = [PROMPT_TEMPLATE.format(instruction=r["question"]) for r in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg["train"]["max_seq_len"]).to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"],
                # 必须显式传 attention_mask：pad_token 被设成 eos 时 transformers 无法从
                # input_ids 推断掩码，会把左填充当作正文读入。
                attention_mask=enc["attention_mask"],
                max_new_tokens=ecfg["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_text = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for r, txt in zip(chunk, gen_text):
            pred = extract_answer(txt)
            gold = gold_answer(r["answer"])
            ok = pred is not None and gold is not None and pred == gold
            correct += int(ok)
            if len(samples) < 20:
                samples.append({"question": r["question"][:200], "pred": pred, "gold": gold, "ok": ok})
        done = min(start + bs, len(rows))
        pbar.set_postfix(acc=f"{correct / done:.4f}")
    pbar.close()

    acc = correct / max(len(rows), 1)
    print(f"GSM8K 准确率（{args.method}）= {acc:.4f}  ({correct}/{len(rows)})")

    metrics_dir = abspath(cfg["paths"]["output_dir"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"eval_{args.method}.json"
    payload = {
        "experiment": f"M1/eval_gsm8k ({args.method})",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"method": args.method, "num_eval": len(rows), "max_new_tokens": ecfg["max_new_tokens"]},
        "results": {"accuracy": acc, "correct": correct, "total": len(rows), "samples": samples},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 metrics：{out_path}")


if __name__ == "__main__":
    main()
