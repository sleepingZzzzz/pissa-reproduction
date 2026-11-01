import json

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from scripts.m0 import mnist_toy as m


class FakeMNIST(Dataset):
    def __init__(self, n=30):
        self.targets = torch.arange(n) % 10
        self.x = torch.randn(n, 1, 28, 28, generator=torch.Generator().manual_seed(1))
        self.x[:, 0, 0, 0] = torch.arange(n).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.x[index], int(self.targets[index])


def test_batch_pairing_crosses_iterator_boundary():
    dataset = FakeMNIST(10)
    def sequence(seed):
        stream = m.batch_stream(m.make_loader(dataset, 4, seed, True))
        return [next(stream)[0][:, 0, 0, 0].tolist() for _ in range(8)]
    assert sequence(5) == sequence(5)
    assert sequence(5) != sequence(6)  # negative control
    assert sum(map(len, sequence(5)[:3])) == 10  # retain short last batch


def test_injected_loaders_use_labels_and_official_test_role():
    loaders = m.get_loaders(4, seed=2, train_dataset=FakeMNIST(), test_dataset=FakeMNIST(10))
    assert len(loaders["odd"].dataset) == 15
    assert len(loaders["even"].dataset) == 15
    assert len(loaders["test"].dataset) == 5
    assert all(y % 2 == 0 for _, y in loaders["test"].dataset)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_lora_alpha_contract(alpha):
    with pytest.raises(ValueError):
        m.LoRALinear(nn.Linear(8, 8), 2, alpha=alpha)


def test_lora_gaussian_scale(monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda *shape, **kwargs: torch.ones(*shape, **kwargs))
    layer = m.LoRALinear(nn.Linear(8, 8), 4)
    assert torch.all(layer.lora_A == .25)
    assert torch.count_nonzero(layer.lora_B) == 0


@pytest.mark.parametrize("method", ["fc_full", "full_ft", "lora", "pissa"])
def test_method_contract_detects_bad_trainability(method):
    base = m.MLP(hidden=8)
    probe = torch.randn(3, 1, 28, 28)
    model, info = m.prepare_method(base, method, 2, probe)
    expected = {"fc_full": 72, "full_ft": sum(p.numel() for p in base.parameters()),
                "lora": 32, "pissa": 32}
    assert info["trainable_parameters"] == expected[method]
    assert info["initial_logits_close"]
    if method != "full_ft":
        model.enc.weight.requires_grad_(True)
        with pytest.raises(ValueError):
            m.validate_initialization(base, model, method, 2, probe)


def test_tiny_injected_pipeline_and_output(tmp_path):
    code = m.main(["--epochs", "1", "--ft-steps", "4", "--batch", "4", "--hidden", "8",
                   "--r", "2", "--seeds", "0", "1", "--eval-every", "2",
                   "--output", str(tmp_path), "--device", "cpu"],
                  datasets=(FakeMNIST(), FakeMNIST(10)))
    assert code == 0
    payload = json.loads((tmp_path / "metrics/mnist.json").read_text())
    assert payload["status"] == "passed"
    assert payload["config"]["data_source"] == "injected_test_fixture"
    assert len(payload["results"]["runs"]) == 2
    for run in payload["results"]["runs"]:
        assert set(run["methods"]) == {"fc_full", "full_ft", "lora", "pissa"}
        for result in run["methods"].values():
            assert len(result["loss_curve"]) == 4
            assert result["frozen_unchanged"]
            assert result["evaluations"][-1]["samples"] == 5
    assert (tmp_path / "figures/mnist_toy_convergence.png").exists()


def test_lora_peft_reference():
    result = m.check_lora_peft()
    if result.get("skipped"):
        pytest.skip(result["reason"])
    assert result["passed"], result


def test_frozen_mutation_detected():
    base = m.MLP(hidden=8)
    model, _ = m.prepare_method(base, "lora", 2, torch.randn(2, 1, 28, 28))
    before = m.frozen_state(model)
    with torch.no_grad():
        model.enc.weight.add_(.1)
    with pytest.raises(ValueError, match="frozen"):
        m.validate_frozen(model, before)


def test_evaluation_weights_short_last_batch():
    model = m.MLP(hidden=8)
    dataset = FakeMNIST(7)
    one = m.evaluate(model, m.make_loader(dataset, 7, 0, False), "cpu")
    split = m.evaluate(model, m.make_loader(dataset, 3, 0, False), "cpu")
    assert one["loss"] == pytest.approx(split["loss"], rel=1e-6)
    assert one["accuracy"] == split["accuracy"]


def test_toy_parameter_error_is_64():
    with pytest.raises(SystemExit) as exc:
        m.main(["--r", "0"])
    assert exc.value.code == 64
