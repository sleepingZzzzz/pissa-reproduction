import json

import pytest
import torch
from torch import nn

from pissa import PiSSALinear, pissa_init
from scripts.m0 import verify_init as v


def test_independent_fixture_separates_correct_and_wrong_before_changes():
    """Pre-implementation sensitivity check, without any new script helper."""
    w = torch.diag(torch.tensor([20., 15., 10., 1., .5, .1]))
    expected = torch.diag(torch.tensor([20., 15., 10., 0., 0., 0.]))
    correct = pissa_init(w, 3)
    wrong = torch.diag(torch.tensor([20., 15., 0., 1., 0., 0.]))
    accepts = lambda delta: torch.allclose(delta, expected, atol=1e-5, rtol=1e-5)
    assert accepts(correct.lora_B @ correct.lora_A)
    assert not accepts(wrong)
    assert torch.linalg.matrix_norm(w - wrong, ord=2) > 9 * torch.linalg.matrix_norm(w - expected, ord=2)


def test_known_oracle_rejects_wrong_boundary_component():
    w, expected, s = v.known_spectrum(20, 16, 4, "gapped", 12)
    init = pissa_init(w, 4)
    assert v.assess_decomposition(w, init.lora_B @ init.lora_A, init.residual, 4,
                                 expected_s=s, expected_delta=expected)["passed"]
    u, values, vh = torch.linalg.svd(w.double(), full_matrices=False)
    indices = [0, 1, 2, 4]  # deliberately replace the fourth by the fifth
    bad = ((u[:, indices] * values[indices]) @ vh[indices]).float()
    result = v.assess_decomposition(w, bad, w - bad, 4,
                                    expected_s=s, expected_delta=expected)
    assert not result["passed"]


def test_validator_rejects_fake_residual_and_nan():
    w, expected, s = v.known_spectrum(12, 10, 3, "gapped", 1)
    assert not v.assess_decomposition(w, expected.float(), torch.zeros_like(w), 3,
                                      expected_s=s, expected_delta=expected)["passed"]
    bad = expected.float().clone()
    bad[0, 0] = float("nan")
    assert not v.assess_decomposition(w, bad, w - bad, 3)["passed"]


def test_known_check_rejects_mutated_initializer(monkeypatch):
    """Integration mutation: even a self-consistent W-residual cannot hide bad BA."""
    original = v.pissa_init
    def wrong(w, r, *args, **kwargs):
        result = original(w, r, *args, **kwargs)
        u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
        ids = list(range(r-1))+[r]
        result.lora_B = u[:, ids]*s[ids].sqrt()
        result.lora_A = s[ids].sqrt()[:, None]*vh[ids]
        result.residual = w-result.lora_B @ result.lora_A
        return result
    monkeypatch.setattr(v, "pissa_init", wrong)
    assert not v.check_svd_property(16, 20, 3)["passed"]


def test_reconstruction_rejects_mutated_forward(monkeypatch):
    original = PiSSALinear.forward
    monkeypatch.setattr(PiSSALinear, "forward", lambda self, x: original(self, x)+.1)
    assert not v.check_reconstruction(12, 10, 3)["passed"]


def test_repeated_boundary_has_no_unique_principal_requirement():
    w = torch.eye(4)
    # Either of two different rank-2 projections has the optimal residual.
    for ids in ([0, 1], [2, 3]):
        delta = torch.zeros_like(w)
        delta[ids, ids] = 1
        result = v.assess_decomposition(w, delta, w-delta, 2,
            expected_s=torch.ones(4), expected_delta=torch.diag(torch.tensor([1., 1., 0., 0.])))
        assert result["passed"]
        assert not result["direction_comparison_stable"]


@pytest.mark.parametrize("shape,r,scale", [((8, 5), 1, 1.), ((5, 8), 5, 1.),
                                           ((6, 6), 6, 0.), ((7, 5), 5, 1e-8)])
def test_zero_full_rank_and_rectangular(shape, r, scale):
    torch.manual_seed(2)
    w = torch.randn(*shape) * scale
    init = pissa_init(w, r)
    assert v.assess_decomposition(w, init.lora_B @ init.lora_A, init.residual, r)["passed"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_from_linear_preserves_dtype_and_device(dtype):
    linear = nn.Linear(8, 6, dtype=dtype)
    layer = PiSSALinear.from_linear(linear, 2)
    assert layer.lora_A.dtype == linear.weight.dtype
    assert layer.residual.device == linear.weight.device
    assert layer.residual.dtype == linear.weight.dtype
    assert layer.bias.dtype == linear.bias.dtype


def test_alpha_negative_contract():
    with pytest.raises(ValueError):
        PiSSALinear.from_linear(nn.Linear(8, 8), 2, alpha=4)


def test_q_default_unchanged_and_rank_stays_r(monkeypatch):
    calls = []
    original = torch.svd_lowrank

    def spy(w, q, niter):
        calls.append(q)
        return original(w, q=q, niter=niter)

    monkeypatch.setattr(torch, "svd_lowrank", spy)
    w = torch.randn(20, 16)
    torch.manual_seed(7)
    implicit = pissa_init(w, 3, niter=2)
    torch.manual_seed(7)
    explicit = pissa_init(w, 3, niter=2, q=11)
    assert calls == [11, 11]
    assert torch.equal(implicit.lora_A, explicit.lora_A)
    assert torch.equal(implicit.residual, explicit.residual)
    assert pissa_init(w, 3, niter=2, q=3).lora_B.shape == (20, 3)


@pytest.mark.parametrize("q", [1, 17, True, 3.5])
def test_invalid_q(q):
    with pytest.raises(ValueError):
        pissa_init(torch.randn(20, 16), 3, niter=2, q=q)


@pytest.mark.parametrize("kind,code,status", [("pass", 0, "passed"),
    ("fail", 1, "failed"), ("skip", 2, "partial"), ("error", 1, "error")])
def test_incremental_status_and_exit(tmp_path, kind, code, status):
    path = tmp_path / "metrics" / "verify.json"

    def check():
        disk = json.loads(path.read_text())
        assert disk["status"] == "running" and disk["exit_code"] is None
        if kind == "error":
            raise RuntimeError("deliberate failure")
        if kind == "skip":
            return {"skipped": True, "reason": "deliberate skip"}
        return {"passed": kind == "pass"}

    assert v.run_checks(path, {}, {"probe": check}) == code
    disk = json.loads(path.read_text())
    assert disk["status"] == status and disk["exit_code"] == code
    assert disk["all_pass"] == (code == 0)


def test_error_has_priority_over_skip(tmp_path):
    assert v.run_checks(tmp_path / "result.json", {}, {
        "skip": lambda: {"skipped": True}, "fail": lambda: {"passed": False}}) == 1


def test_atomic_save_failure_not_hidden(tmp_path, monkeypatch):
    def fail(*args):
        raise PermissionError("occupied")
    monkeypatch.setattr(v.os, "replace", fail)
    with pytest.raises(PermissionError):
        v.atomic_json(tmp_path / "x.json", {"status": "running"})


def test_cli_partial_and_parameter_error(tmp_path):
    assert v.main(["--in-features", "12", "--out-features", "10", "--r", "2",
                   "--seeds", "0", "--niters", "2", "--skip-peft",
                   "--output", str(tmp_path)]) == 2
    with pytest.raises(SystemExit) as exc:
        v.main(["--r", "0"])
    assert exc.value.code == 64


def test_fast_quality_covers_spectra():
    result = v.check_fast_quality(16, 20, 3, [2], [0, 1], "cpu")
    assert result["passed"]
    assert {row["spectrum"] for row in result["samples"]} == {"gaussian", "decaying", "gapped"}
    assert {row["q"] for row in result["samples"]} == {3, 11}


def test_fast_peft_rng_isolated():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    result = v.check_against_peft(3, niter=2)
    assert torch.equal(before, torch.random.get_rng_state())
    if result.get("skipped"):
        pytest.skip(result["reason"])
    assert result["passed"], result


def test_peft_rank_can_exceed_old_128_limit():
    result = v.check_against_peft(129)
    if result.get("skipped"):
        pytest.skip(result["reason"])
    assert result["passed"], result
    assert all(sample["shape"][0] >= 129 for sample in result["samples"])


def test_q_is_only_for_fast_and_default_is_clipped():
    w = torch.randn(10, 8)
    with pytest.raises(ValueError):
        pissa_init(w, 3, q=3)
    torch.manual_seed(12)
    implicit = pissa_init(w, 5, niter=2)
    torch.manual_seed(12)
    explicit = pissa_init(w, 5, niter=2, q=8)
    assert torch.equal(implicit.residual, explicit.residual)
