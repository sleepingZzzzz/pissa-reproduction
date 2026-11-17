"""JSONL 读取、官方协议的指令分词、截断分类与训练批次右填充。

分词协议对齐官方 PiSSA 训练入口（9633d875 的 pissa.py，见 HANDOFF P0/P3）：
    - 提示以 ``### Response:`` 结尾，不追加空白或换行；
    - 目标为 ``回答 + "\\n" + eos``，直接拼在提示之后；其中 eos 以 token id 追加（不是文本 ``</s>``，
      见 build_texts）；
    - 提示与目标拼成整串后联合分词并整体截断到 max_len；
    - 提示部分的标签屏蔽长度取"提示单独分词"的长度（同样截断到 max_len）。
"""
from __future__ import annotations

import json
from pathlib import Path
import torch

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

# 截断分类的取值（T4）。
TRUNC_NONE = "untruncated"
TRUNC_EOS_ONLY = "eos_only_lost"
TRUNC_BODY_PARTIAL = "response_body_partially_lost"
TRUNC_ALL_LOST_PROMPT_OK = "response_all_lost_prompt_intact"
TRUNC_ALL_LOST_PROMPT_CUT = "response_all_lost_prompt_truncated"
TRUNC_CATEGORIES = (
    TRUNC_NONE,
    TRUNC_EOS_ONLY,
    TRUNC_BODY_PARTIAL,
    TRUNC_ALL_LOST_PROMPT_OK,
    TRUNC_ALL_LOST_PROMPT_CUT,
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


def _tokenize(tokenizer, text: str, max_len: int | None) -> list[int]:
    """启用特殊 token 分词；max_len 非 None 时右侧截断到该长度。"""
    if max_len is None:
        return list(tokenizer(text, add_special_tokens=True)["input_ids"])
    return list(
        tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    )


def build_texts(tokenizer, instruction: str, response: str) -> tuple[str, str]:
    """返回 (提示, 目标正文)：提示以 ``### Response:`` 结尾，目标正文为 ``回答\\n``（不含 EOS）。

    EOS 不拼进文本：transformers 5.x 的 Llama 分词器会把文本 ``</s>`` 切成 ``</``、``s``、``>``
    三个普通 token（模型因此学不到真正的 EOS），所以 EOS 由 :func:`_joint_ids` 以 token id 追加。
    """
    source = PROMPT_TEMPLATE.format(instruction=instruction)
    return source, f"{response}\n"


def _joint_ids(tokenizer, source: str, body: str, max_len: int | None) -> list[int]:
    """整串联合分词（启用 BOS），再追加真正的 EOS token id；max_len 非 None 时整体右侧截断。

    与官方 transformers 4.x 下对 ``source + response + "\\n" + eos_token`` 分词的结果一致。
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("分词器没有 eos_token_id，无法追加 EOS。")
    ids = _tokenize(tokenizer, source + body, None) + [int(eos_id)]
    return ids if max_len is None else ids[:max_len]


def tokenize_example(tokenizer, instruction: str, response: str, max_len: int, response_only: bool):
    """按官方协议构造单条训练样本：整串联合分词、整体截断，提示标签屏蔽。

    Args:
        tokenizer: 提供 eos_token_id 的分词器；分词启用特殊 token（含 BOS）。
        instruction: 插入 PROMPT_TEMPLATE 的问题文本。
        response: 回答文本；序列为 提示 + response + 换行 + EOS（EOS 以 id 追加）。
        max_len: 整串保留的最大 token 数（提示单独分词时同样以此截断）。
        response_only: 为 True 时前 len(提示单独分词) 个标签设为 -100。

    Returns:
        包含等长整数列表 input_ids、labels 的字典，尚未填充或转换为张量。
        截断可能移除回答或 EOS；本函数不丢弃全为 -100 的样本。
    """
    source, body = build_texts(tokenizer, instruction, response)
    input_ids = _joint_ids(tokenizer, source, body, max_len)
    if response_only:
        source_len = len(_tokenize(tokenizer, source, max_len))
        labels = list(input_ids)
        labels[:source_len] = [-100] * min(source_len, len(labels))
    else:
        labels = list(input_ids)
    return {"input_ids": input_ids, "labels": labels}


def valid_target_count(labels: list[int]) -> int:
    """返回因果移位后参与预测的有效目标数：labels[1:] 中非 -100 的个数。"""
    return sum(1 for v in labels[1:] if v != -100)


def classify_truncation(tokenizer, instruction: str, response: str, max_len: int) -> dict:
    """按训练实际使用的规则对单条样本做截断分类（T4）。

    分类互斥，依次为：未截断；仅丢 EOS；回答正文部分丢失（仍有有效目标）；
    回答全丢但提示完整；提示也被截断且回答全丢。

    Returns:
        字典：category、full_len（未截断长度）、source_len（提示长度，未截断）、
        kept_len（截断后长度）、valid_targets（移位后有效目标数，训练真实处理规则）。
    """
    source, body = build_texts(tokenizer, instruction, response)
    full_ids = _joint_ids(tokenizer, source, body, None)
    source_len = len(_tokenize(tokenizer, source, None))
    example = tokenize_example(tokenizer, instruction, response, max_len, True)
    kept_len = len(example["input_ids"])
    valid = valid_target_count(example["labels"])
    cut = len(full_ids) - kept_len
    if cut <= 0:
        category = TRUNC_NONE
    elif valid == 0:
        category = TRUNC_ALL_LOST_PROMPT_OK if source_len <= max_len else TRUNC_ALL_LOST_PROMPT_CUT
    elif cut == 1:
        category = TRUNC_EOS_ONLY
    else:
        category = TRUNC_BODY_PARTIAL
    return {
        "category": category,
        "full_len": len(full_ids),
        "source_len": source_len,
        "kept_len": kept_len,
        "valid_targets": valid,
    }


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
            （官方用 ne(pad_id) 作掩码，会把末尾真实 EOS 的掩码置 0；该位置之后没有
            预测目标，对损失无影响，本项目保留"真实位置为 1"的约定。）

        Raises:
            ValueError: 某条样本的 input_ids 与 labels 长度不一致。
        """
        for b in batch:
            if len(b["input_ids"]) != len(b["labels"]):
                raise ValueError(
                    f"input_ids 与 labels 长度不一致：{len(b['input_ids'])} vs {len(b['labels'])}"
                )
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
