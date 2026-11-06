"""JSONL 读取、指令分词与训练批次右填充。"""
from __future__ import annotations

import json
from pathlib import Path
import torch

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def read_jsonl(path: str | Path) -> list[dict]:
    """按文件顺序解析 JSONL 的所有非空行。

    Args:
        path: UTF-8 JSONL 文件路径。

    Returns:
        每个非空行解析所得对象组成的列表；不校验字段。
    """
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_example(tokenizer, instruction: str, response: str, max_len: int, response_only: bool):
    """套用指令模板，拼接回答并右侧截断，构造单条训练样本。

    Args:
        tokenizer: 提供 eos_token 的分词器。提示启用特殊 token，回答不重复添加。
        instruction: 插入 PROMPT_TEMPLATE 的问题文本。
        response: 回答文本；分词前拼接 eos_token。
        max_len: 拼接序列保留的最大 token 数。
        response_only: 为 True 时将提示部分的标签设为 -100。

    Returns:
        包含等长整数列表 input_ids、labels 的字典，尚未填充或转换为张量。
        截断可能移除回答或 EOS；本函数不丢弃全为 -100 的样本。
    """
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    resp_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + resp_ids)[:max_len]
    if response_only:
        labels = ([-100] * len(prompt_ids) + resp_ids)[:max_len]
    else:
        labels = list(input_ids)
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    """将变长样本右填充到当前批次的最长长度，不依赖分词器填充方向。"""
    def __init__(self, pad_id: int):
        """保存 input_ids 填充所用的 token ID。

        Args:
            pad_id: 填充位置的 ID；可以与真实 EOS 的 ID 相同。
        """
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """返回右填充后的 input_ids、labels 和 attention_mask。

        Args:
            batch: 非空样本列表；每条含等长的 input_ids 和 labels 整数列表。

        Returns:
            三个形状均为 (N, L) 的 CPU torch.long 张量。
            L 为当前批次最长长度；填充 ID 为 pad_id，标签为 -100，掩码为 0。
            真实位置的掩码为 1，因此真实 EOS 不会因与 pad_id 相同而被屏蔽。
        """
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
