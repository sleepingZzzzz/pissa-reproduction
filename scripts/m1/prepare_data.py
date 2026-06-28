"""[M1] 下载并处理 MetaMathQA（训练）与 GSM8K（评测），落地为本地 jsonl。

train/eval 脚本只读这两个 jsonl，与下载源解耦（modelscope 或 hf）。

用法：
    python scripts/m1/prepare_data.py --config configs/m1_llama2_7b.yaml
    python scripts/m1/prepare_data.py --config configs/m1_llama2_7b.yaml --limit 2000  # 冒烟
"""

from __future__ import annotations

import argparse
import json

from common import abspath, load_config


def _load_metamath(cfg: dict):
    dcfg = cfg["data"]
    if dcfg["source"] == "modelscope":
        from modelscope.msdatasets import MsDataset

        print(f"[modelscope] 加载 MetaMathQA：{dcfg['metamath_ms_repo']}")
        ds = MsDataset.load(dcfg["metamath_ms_repo"], split="train", trust_remote_code=True)
    else:
        from datasets import load_dataset

        print(f"[hf] 加载 MetaMathQA：{dcfg['metamath_hf_repo']}")
        ds = load_dataset(dcfg["metamath_hf_repo"], split="train")
    return ds


def _load_gsm8k(cfg: dict):
    dcfg = cfg["data"]
    if dcfg["source"] == "modelscope":
        from modelscope.msdatasets import MsDataset

        print(f"[modelscope] 加载 GSM8K：{dcfg['gsm8k_ms_repo']} / {dcfg['gsm8k_subset']}")
        ds = MsDataset.load(
            dcfg["gsm8k_ms_repo"],
            subset_name=dcfg["gsm8k_subset"],
            split="test",
            trust_remote_code=True,
        )
    else:
        from datasets import load_dataset

        print(f"[hf] 加载 GSM8K：{dcfg['gsm8k_hf_repo']} / {dcfg['gsm8k_subset']}")
        ds = load_dataset(dcfg["gsm8k_hf_repo"], dcfg["gsm8k_subset"], split="test")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="[M1] 准备 MetaMathQA + GSM8K")
    parser.add_argument("--config", default="configs/m1_llama2_7b.yaml")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条（冒烟测试）")
    args = parser.parse_args()

    cfg = load_config(abspath(args.config))
    dcfg = cfg["data"]

    # ---- 训练集 ----
    n_limit = args.limit if args.limit is not None else dcfg["num_train_samples"]
    ds = _load_metamath(cfg)
    train_path = abspath(dcfg["train_jsonl"])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if n_limit is not None and i >= n_limit:
                break
            # MetaMathQA 字段：query / response。
            instruction = row.get("query") or row.get("question") or row.get("instruction")
            response = row.get("response") or row.get("answer")
            if not instruction or not response:
                continue
            f.write(json.dumps({"instruction": instruction, "response": response}, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"训练集已写入 {train_path}（{n_written} 条）")

    # ---- 评测集 ----
    gsm = _load_gsm8k(cfg)
    gsm_path = abspath(dcfg["gsm8k_jsonl"])
    n_eval = 0
    with open(gsm_path, "w", encoding="utf-8") as f:
        for row in gsm:
            question = row.get("question")
            answer = row.get("answer")
            if not question or not answer:
                continue
            f.write(json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + "\n")
            n_eval += 1
    print(f"GSM8K 评测集已写入 {gsm_path}（{n_eval} 条）")


if __name__ == "__main__":
    main()
