"""Supplementary diagnostics: offline, small fixtures, no formal output writes."""
import copy
import json

import pytest
import torch

from scripts.m0 import mnist_toy as m
from scripts.m0 import verify_init as v
from test_m0_toy import FakeMNIST


def test_fast_sensitivity_reports_bad_settings_without_changing_limits():
    before = dict(v.FAST_LIMITS)
    report = v.check_fast_sensitivity(24, 32, 3, [0, 1])
    assert report["passed"] and report["diagnostic_only"]
    assert report["retention_informed_by_prior_results"]
    assert {x["spectrum"] for x in report["samples"]} == set(before)
    assert {x["niter"] for x in report["samples"]} == {0}
    assert {x["q"] for x in report["samples"]} == {3, 11}
    assert all("within_existing_limit" in x for x in report["samples"])
    assert all(x["quality_ratio"] is not None for x in report["samples"])
    assert v.FAST_LIMITS == before


def test_bad_diagnostic_quality_is_not_a_new_failure_criterion(monkeypatch):
    original = v.pissa_init
    def bad(weight, r, *args, **kwargs):
        result = original(weight, r, *args, **kwargs)
        result.lora_A.zero_()
        result.lora_B.zero_()
        result.residual = weight.clone()
        return result
    monkeypatch.setattr(v, "pissa_init", bad)
    result = v.check_fast_sensitivity(24, 32, 3, [0])
    assert result["passed"]
    assert any(not row["within_existing_limit"] for row in result["samples"])


@pytest.mark.parametrize("niter", [None, 2])
def test_peft_includes_small_rank_fraction(niter):
    result = v.check_against_peft(3, niter=niter)
    if result.get("skipped"):
        pytest.skip(result["reason"])
    low = [x for x in result["samples"] if x["regime"] == "low_rank_fraction"]
    assert result["passed"]
    assert {(x["shape"][0], x["r"]) for x in low} == {(256, 16)}
    assert {x["spectrum"] for x in low} == {"gapped", "gaussian"}


def test_scan_split_is_fixed_disjoint_and_does_not_load_test(monkeypatch):
    data = FakeMNIST(100)
    before = torch.random.get_rng_state().clone()
    first = m.get_scan_loaders(8, seed=1, split_seed=17, train_dataset=data)
    second = m.get_scan_loaders(8, seed=2, split_seed=17, train_dataset=data)
    assert torch.equal(before, torch.random.get_rng_state())
    assert "test" not in first
    train_ids = set(first["even"].dataset.indices)
    val_ids = set(first["validation"].dataset.indices)
    assert not train_ids & val_ids
    assert train_ids | val_ids == set(range(0, 100, 2))
    assert len(val_ids) == 5
    assert first["validation"].dataset.indices == second["validation"].dataset.indices
    assert first["split"] == second["split"]


@pytest.mark.parametrize("method", m.METHODS)
@pytest.mark.parametrize("steps", [1, 7])
def test_displacement_matches_effective_weight_and_does_not_change_training(method, steps):
    with v.seeded(7):
        base = m.MLP(8)
        probe = torch.randn(3, 1, 28, 28)
        model, _ = m.prepare_method(base, method, 2, probe)
    assert torch.allclose(m.effective_fc_weight(model), base.fc.weight, atol=1e-6)
    data = FakeMNIST(20)
    initial = m.effective_fc_weight(model).clone()
    twin = copy.deepcopy(model)
    loader = lambda: m.make_loader(data, 4, 99, True)
    evaluation = m.make_loader(data, 4, 0, False)
    state = torch.random.get_rng_state().clone()
    measured = m.finetune(model, loader(), steps, .001, "cpu", evaluation, 1, 1)
    unmeasured = m.finetune(twin, loader(), steps, .001, "cpu", evaluation, 1, 1,
                            record_displacement=False)
    assert torch.equal(state, torch.random.get_rng_state())
    expected = float(torch.linalg.vector_norm(m.effective_fc_weight(model)-initial))
    assert len(measured["step_displacements"]) == steps
    if steps == 1:
        assert measured["step_displacements"] == pytest.approx([expected])
    else:
        assert sum(measured["step_displacements"])+1e-6 >= expected
    assert expected > 0
    assert measured["loss_curve"] == unmeasured["loss_curve"]
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), twin.state_dict().values()))


def test_scan_grid_is_not_a_cross_product():
    grid = m.scan_grid([.0003, .001, .003], [8, 32, 128, 256], 8, .001)
    assert len(grid) == 18  # 12 lr cells + 6 additional rank cells
    assert all(x["r"] == 8 or x["lr"] == .001 for x in grid)
    assert all(x["method"] in ("lora", "pissa") for x in grid if "rank" in x["axes"])


def test_small_scan_pretrains_once_and_never_evaluates_official_test(tmp_path, monkeypatch):
    data = FakeMNIST(100)
    real_pretrain, real_evaluate = m.pretrain, m.evaluate
    pretrain_calls, seen_validation = [], []
    def pretrain(*args, **kwargs):
        pretrain_calls.append(1)
        return real_pretrain(*args, **kwargs)
    def evaluate(model, loader, device):
        seen_validation.append(set(loader.dataset.indices))
        return real_evaluate(model, loader, device)
    monkeypatch.setattr(m, "pretrain", pretrain)
    monkeypatch.setattr(m, "evaluate", evaluate)
    assert m.main(["--scan", "--epochs", "1", "--ft-steps", "4", "--hidden", "8",
                   "--r", "2", "--seeds", "0", "1", "--batch", "8", "--device", "cpu",
                   "--scan-lrs", ".001", ".003", "--scan-ranks", "2", "4", "8",
                   "--scan-checkpoints", "2", "4", "--output", str(tmp_path)],
                  datasets=(data, None)) == 0
    assert len(pretrain_calls) == 2
    assert all(ids == seen_validation[0] for ids in seen_validation)
    payload = json.loads((tmp_path / "metrics/mnist_scan.json").read_text())
    assert payload["status"] == "passed" and payload["config"]["mode"] == "diagnostic_scan"
    assert payload["protocol"]["evaluation_split"] == "training_even_validation_only"
    assert payload["results"]["comparisons"]["best_lr_by_method"]
    assert len(payload["results"]["table"]) == 2 * 12 * 2
    assert (tmp_path / "metrics/mnist_scan.csv").exists()
    assert not (tmp_path / "metrics/mnist.json").exists()


def test_scan_rejects_formal_output_directory():
    with pytest.raises(SystemExit) as exc:
        m.main(["--scan"])
    assert exc.value.code == 64


def test_best_lr_uses_across_seed_mean_not_per_seed_winners():
    protocol = {"seeds": [0, 1], "checkpoints": [1, 2], "base_r": 2,
                "base_lr": .001, "lrs": [.001, .003], "ranks": [2]}
    table = [{"seed": seed, "method": method, "lr": lr, "r": 2, "step": step,
              "validation_loss": ([0., 10.][seed] if lr == .001 else 4.)}
             for seed in [0, 1] for method in m.METHODS for lr in [.001, .003] for step in [1, 2]]
    result = m.scan_comparisons(table, protocol)
    assert all(x["lr"] == .003 for x in result["best_lr_by_method"].values())
    assert result["best_lr_by_method"]["pissa"]["per_seed_final_validation_loss"] == {"0": 4., "1": 4.}
