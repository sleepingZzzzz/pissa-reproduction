"""回退评分 last_number_v1 与 run_exp 的 base 布局。"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gs = _load("gsm8k_score", "src/scripts/m1/gsm8k_score.py")
run_exp = _load("run_exp_base", "src/scripts/m1/run_exp.py")


def test_last_number_taken_without_marker():
    ok, cand = gs.score_fallback("She makes 9 * 2 = $18 every day.", 18.0)
    assert ok and cand == "18"


def test_marker_not_required_and_last_number_wins():
    ok, cand = gs.score_fallback("The answer is: 5\nQuestion 7", 5.0)
    assert not ok and cand == "7"  # 回退评分只看最后一个数字，不看标记


def test_thousand_separator_decimal_and_negative():
    assert gs.score_fallback("total is 70,000", 70000.0)[0]
    assert gs.score_fallback("so 18.50 dollars", 18.5)[0]
    assert gs.score_fallback("loss of -39", -39.0)[0]


def test_no_rounding():
    assert not gs.score_fallback("about 18.4", 18.0)[0]


def test_fraction_and_zero_denominator():
    assert gs.score_fallback("answer 3/4", 0.75)[0]
    ok, cand = gs.score_fallback("answer 3/0", 3.0)
    assert not ok and cand == "3/0"


def test_no_number():
    assert gs.score_fallback("no digits here", 1.0) == (False, None)


def test_primary_scorer_unchanged_by_fallback_addition():
    assert gs.score_primary("blah\nThe answer is: 18", 18.0) == (True, 18)
    assert gs.score_primary("so it is 18", 18.0) == (False, None)  # 主评分仍无回退


def test_base_kind_layout_has_no_ckpt():
    cfg = {"experiment": "m2_qwen25_7b_paper512", "model": {"name": "Qwen2.5-7B"}}
    repo = Path("/repo")
    stage, tag, res, ck = run_exp.stage_layout(cfg, "base", repo)
    assert (stage, tag) == ("m2", "m2_Qwen2.5-7B_base")
    assert res == repo / "results/m2/Qwen2.5-7B/base" and ck is None
