"""run_exp --kind table4：M3 Table 4 的产物按 exp 号放进 results/m3/{模型名}/table4。"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("run_exp_t4", ROOT / "src" / "scripts" / "m1" / "run_exp.py")
run_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_exp)

REPO = Path("/repo")


def test_table4_kind_registered_and_has_no_ckpt():
    assert "table4" in run_exp.KINDS
    cfg = {"experiment": "m3_table4_llama3_8b", "model": {"name": "Llama-3-8B"}}
    assert run_exp.stage_layout(cfg, "table4", REPO) == (
        "m3", "m3_Llama-3-8B_table4", REPO / "results/m3/Llama-3-8B/table4", None)


def test_table4_configs_resolve_to_m3_stage():
    from utils import load_config
    for name in ("m3_table4_llama2_7b", "m3_table4_llama3_8b"):
        cfg = load_config(ROOT / "configs" / "paper" / f"{name}.yaml")
        assert run_exp.stage_layout(cfg, "table4", REPO)[0] == "m3"
