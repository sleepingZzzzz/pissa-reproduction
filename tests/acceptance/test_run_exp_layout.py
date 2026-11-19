"""run_exp.stage_layout：m1 保持原布局，m2/m3 的产物按阶段与模型名分目录。"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("run_exp", ROOT / "src" / "scripts" / "m1" / "run_exp.py")
run_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_exp)

REPO = Path("/repo")


def _cfg(exp, model):
    return {"experiment": exp, "model": {"name": model}}


def test_m1_layout_unchanged():
    cfg = _cfg("m1_llama2_7b_paper512", "Llama-2-7b")
    assert run_exp.stage_layout(cfg, "full", REPO) == (
        "m1", "m1_Llama-2-7b", REPO / "results/m1/full", REPO / "checkpoints/full")
    assert run_exp.stage_layout(cfg, "smoke", REPO) == (
        "m1", "m1_Llama-2-7b_smoke", REPO / "results/m1/smoke", REPO / "checkpoints/smoke")
    stage, tag, res, ck = run_exp.stage_layout(cfg, "tokenize", REPO)
    assert (res, ck) == (REPO / "results/m1/tokenize_stats", None)


def test_m2_results_and_full_ckpt_per_model():
    cfg = _cfg("m2_qwen25_7b_paper512", "Qwen2.5-7B")
    assert run_exp.stage_layout(cfg, "full", REPO) == (
        "m2", "m2_Qwen2.5-7B", REPO / "results/m2/Qwen2.5-7B/full", REPO / "checkpoints/m2/Qwen2.5-7B/full")


def test_m2_smoke_results_per_model_ckpt_global():
    cfg = _cfg("m2_gemma_7b_paper512", "Gemma-7B")
    stage, tag, res, ck = run_exp.stage_layout(cfg, "smoke", REPO)
    assert (stage, tag) == ("m2", "m2_Gemma-7B_smoke")
    assert res == REPO / "results/m2/Gemma-7B/smoke"
    assert ck == REPO / "checkpoints/smoke"


def test_missing_experiment_field_falls_back_to_m1():
    stage, tag, res, ck = run_exp.stage_layout({"model": {"name": "Toy"}}, "full", REPO)
    assert (stage, res, ck) == ("m1", REPO / "results/m1/full", REPO / "checkpoints/full")


def test_unknown_stage_rejected():
    with pytest.raises(SystemExit):
        run_exp.stage_layout(_cfg("foo_bar", "X"), "full", REPO)
