"""训练序列末尾必须是真正的 EOS token id（新增文件）。

背景：transformers 5.x 把文本 ``</s>`` 切成 ``</``、``s``、``>``，模型因此学不到 EOS。
"""
from pathlib import Path

import pytest

from scripts import data_ops as data


class Stub:
    eos_token = "</s>"
    eos_token_id = 2

    def __call__(self, text, add_special_tokens, **kw):
        # BOS + 每个字符一个 id；文本里的 "</s>" 若被拼进去会被这里当作 4 个普通字符
        return {"input_ids": [1] + [100 + ord(c) % 50 for c in text]}


def test_eos_is_appended_as_id_and_not_as_text():
    ids = data.tokenize_example(Stub(), "q", "ab", 10_000, True)["input_ids"]
    assert ids[-1] == 2
    body = Stub()(data.PROMPT_TEMPLATE.format(instruction="q") + "ab\n", True)["input_ids"]
    assert ids == body + [2]  # 没有把 "</s>" 当成文本拼进去


def test_truncation_applies_to_whole_sequence_including_eos():
    full = data.tokenize_example(Stub(), "q", "ab", 10_000, True)["input_ids"]
    cut = data.tokenize_example(Stub(), "q", "ab", len(full) - 1, True)["input_ids"]
    assert cut == full[:-1] and cut[-1] != 2
    info = data.classify_truncation(Stub(), "q", "ab", len(full) - 1)
    assert info["category"] == data.TRUNC_EOS_ONLY and info["full_len"] == len(full)


def test_real_llama_tokenizer_ends_with_eos_id_2():
    snaps = Path(__file__).resolve().parents[2] / "data/m1/models/models/modelscope--Llama-2-7b-ms/snapshots/master"
    if not snaps.exists():
        pytest.skip("本地没有 LLaMA-2 分词器")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snaps))
    ids = data.tokenize_example(tok, "What is 1+1?", "It is 2.\nThe answer is: 2", 512, True)["input_ids"]
    assert ids[-1] == tok.eos_token_id == 2
    assert tok.convert_ids_to_tokens(ids[-2]) == "<0x0A>"
