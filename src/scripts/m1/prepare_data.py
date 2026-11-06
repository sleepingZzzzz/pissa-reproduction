"""[M1] 下载并处理 MetaMathQA（训练）与 GSM8K（评测），落地为本地 jsonl。

train/eval 脚本只读这两个 jsonl，与下载源解耦（modelscope 或 hf）。

用法：
    python src/scripts/m1/prepare_data.py --config configs/m1_llama2_7b.yaml
    python src/scripts/m1/prepare_data.py --config configs/m1_llama2_7b.yaml --limit 2000  # 冒烟
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接执行文件时，将 src 加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

from utils import abspath, load_config


def _load_metamath(cfg: dict):
    """按 data.source 加载 MetaMathQA 的 train split，返回可逐条迭代的数据集。

    Args:
        cfg: 含数据来源和仓库名称的配置；modelscope 走 MsDataset，其余走 HF。

    可能联网下载或使用本地缓存，不在这里截取样本或统一字段。
    """
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
    """按数据来源与子集名称加载 GSM8K test，返回可逐条迭代的数据集。

    Args:
        cfg: 提供 data.source、数据集仓库及 gsm8k_subset 的配置。

    可能联网下载或使用本地缓存，不在这里限制评测数据量。
    """
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
    """解析配置与 limit，将训练及评测数据写成 JSONL，覆盖已有目标文件。

    训练字段统一为 instruction/response；limit 限制原始遍历条数，而非有效写入数。
    评测字段保留 question/answer，写入全部有效 GSM8K test 样本，不按训练 limit 截取。
    """
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
