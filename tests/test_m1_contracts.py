"""离线核对 M1 拆分前后的数据、参数冻结和适配器行为。"""
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "scripts/m1/common.py"
if LEGACY.exists():
    spec = importlib.util.spec_from_file_location("_legacy_m1_contracts", LEGACY)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    data = ops = utils = common
    LoRALinear = common.LoRALinear
else:
    data = importlib.import_module("scripts.data_ops")
    ops = importlib.import_module("scripts.model_ops")
    utils = importlib.import_module("utils")
    from lora import LoRALinear


class TokenizerStub:
    eos_token = "</s>"

    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return {"input_ids": [1, 10, 11] if add_special_tokens else [20, 21, 2]}


@pytest.mark.parametrize("limit,expected_ids,expected_labels", [
    (20, [1, 10, 11, 20, 21, 2], [-100, -100, -100, 20, 21, 2]),
    (4, [1, 10, 11, 20], [-100, -100, -100, 20]),
    (2, [1, 10], [-100, -100]),
])
def test_response_mask_and_prefix_truncation_contract(limit, expected_ids, expected_labels):
    tokenizer = TokenizerStub()
    result = data.tokenize_example(tokenizer, "question", "answer", limit, True)
    assert result == {"input_ids": expected_ids, "labels": expected_labels}
    assert tokenizer.calls == [
        (data.PROMPT_TEMPLATE.format(instruction="question"), True),
        ("answer</s>", False),
    ]


def test_full_sequence_labels_are_a_separate_list():
    result = data.tokenize_example(TokenizerStub(), "q", "a", 5, False)
    assert result["labels"] == [1, 10, 11, 20, 21]
    assert result["labels"] is not result["input_ids"]


def test_collator_uses_lengths_not_eos_values_or_hardcoded_pad():
    result = data.Collator(2)([
        {"input_ids": [10, 20, 2], "labels": [-100, 20, 2]},
        {"input_ids": [2], "labels": [2]},
    ])
    assert torch.equal(result["input_ids"], torch.tensor([[10, 20, 2], [2, 2, 2]]))
    assert torch.equal(result["labels"], torch.tensor([[-100, 20, 2], [2, -100, -100]]))
    assert torch.equal(result["attention_mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]))
    assert all(t.dtype == torch.long and t.shape == (2, 3) for t in result.values())
    alternate = data.Collator(9)([
        {"input_ids": [1, 2], "labels": [1, 2]},
        {"input_ids": [3], "labels": [3]},
    ])
    assert alternate["input_ids"][1].tolist() == [3, 9]


def test_jsonl_preserves_order_and_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"instruction":"甲","response":"乙"}\n\n{"n":2}\n', encoding="utf-8")
    assert data.read_jsonl(path) == [{"instruction": "甲", "response": "乙"}, {"n": 2}]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(9, 4)
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(4, 3, bias=True)
        self.block.v_proj = nn.Linear(4, 3, bias=False)
        self.head = nn.Linear(3, 2)

    def forward(self, ids):
        x = self.embed(ids)
        return self.head(self.block.q_proj(x) + self.block.v_proj(x))


CFG = {"adapter": {"r": 2, "alpha": 2, "target_modules": ["q_proj", "v_proj"], "niter": None}}


@pytest.mark.parametrize("method", ["full", "lora", "pissa"])
def test_method_contract_uses_original_base_denominator_and_freezes_only_as_requested(method):
    torch.manual_seed(42)
    model = TinyModel()
    ids = torch.tensor([[1, 2]])
    original = model(ids).detach()
    assert ops.count_base_params(model) == 71
    stats = ops.apply_adapter(model, method, CFG)
    assert stats["total_params"] == 71
    assert torch.allclose(model(ids), original, atol=2e-6, rtol=2e-6)
    if method == "full":
        assert stats["trainable_params"] == 71
        assert stats["replaced_modules"] == 0
        assert all(p.requires_grad for p in model.parameters())
    else:
        expected = {
            "block.q_proj.lora_A", "block.q_proj.lora_B",
            "block.v_proj.lora_A", "block.v_proj.lora_B",
        }
        assert {n for n, p in model.named_parameters() if p.requires_grad} == expected
        assert stats["trainable_params"] == 28
        assert stats["trainable_pct"] == pytest.approx(100 * 28 / 71)
        assert stats["replaced_modules"] == 2
        assert stats["replaced_by_target"] == {"q_proj": 1, "v_proj": 1}
        with torch.no_grad():
            model.block.q_proj.lora_B.add_(0.125)
        before_merge = model(ids).detach()
        keys = set(model.state_dict())
        assert "block.q_proj.lora_A" in keys
        ops.merge_adapter(model, method)
        assert isinstance(model.block.q_proj, nn.Linear)
        assert torch.allclose(model(ids), before_merge, atol=2e-6, rtol=2e-6)
        assert not any("lora_" in k for k in model.state_dict())
        return
    assert torch.equal(model(ids), original)


def test_lora_kaiming_rng_dtype_buffers_and_initial_identity():
    torch.manual_seed(123)
    base = nn.Linear(4, 3)
    state = torch.random.get_rng_state()
    expected_a = torch.empty(2, 4)
    nn.init.kaiming_uniform_(expected_a, a=5**0.5)
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    layer = LoRALinear(base, r=2, alpha=4)
    assert torch.equal(layer.lora_A, expected_a)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert layer.scaling == 2
    assert torch.count_nonzero(layer.lora_B) == 0
    assert set(dict(layer.named_buffers())) == {"weight", "bias"}
    assert set(dict(layer.named_parameters())) == {"lora_A", "lora_B"}
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert torch.equal(layer(x), base(x))


def test_coverage_zero_total_raises_but_partial_missing_only_warns(capsys):
    with pytest.raises(RuntimeError):
        ops.check_replace_coverage([], ["q_proj"])
    assert ops.check_replace_coverage(["block.q_proj"], ["q_proj", "v_proj"]) == {
        "q_proj": 1, "v_proj": 0,
    }
    assert "v_proj" in capsys.readouterr().out


def test_optimizer_uses_supplied_collection_and_fallback_is_observable(monkeypatch, capsys):
    param = nn.Parameter(torch.ones(3))
    optimizer, kind = ops.build_optimizer([param], 0.002, 0.0, "adamw")
    assert isinstance(optimizer, torch.optim.AdamW)
    assert kind == "adamw"
    assert optimizer.param_groups[0]["params"][0] is param
    assert optimizer.param_groups[0]["lr"] == 0.002
    fake = ModuleType("bitsandbytes")

    class BrokenOptim:
        @staticmethod
        def AdamW8bit(*args, **kwargs):
            raise RuntimeError("contract fallback")

    fake.optim = BrokenOptim
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    optimizer, kind = ops.build_optimizer([param], 0.002, 0.0, "adamw_8bit")
    assert isinstance(optimizer, torch.optim.AdamW) and kind == "adamw"
    assert "contract fallback" in capsys.readouterr().out


def test_local_model_path_wins_without_downloading():
    assert ops.resolve_model_path({"model": {"local_path": "data/fake", "source": "invalid"}}) == str(ROOT / "data/fake")
    assert ops.resolve_model_path({"model": {"source": "hf", "hf_repo": "example/model"}}) == "example/model"
