"""M1 协议验收（新增文件）：调度器、窗口规划、评分、指标定义、配置有效性。"""
import pytest
import torch

from scripts.m1 import gsm8k_score as gs
from scripts.m1 import train as tr
from scripts.m1 import eval_gsm8k as ev


def test_cosine_matches_installed_hf_scheduler():
    total, warm, lr = 97, tr.warmup_steps_for(97, 0.03), 2e-5
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    from transformers import get_cosine_schedule_with_warmup

    sch = get_cosine_schedule_with_warmup(opt, warm, total)
    for step in range(total):
        assert tr.hf_cosine_lr(step, total, warm, lr) == pytest.approx(opt.param_groups[0]["lr"], rel=1e-9, abs=1e-15)
        opt.step()
        sch.step()


def test_warmup_is_ceil():
    assert tr.warmup_steps_for(781, 0.03) == 24  # 23.43 -> 24
    assert tr.warmup_steps_for(100, 0.03) == 3


def test_plan_updates_drops_tail_per_epoch():
    p = tr.plan_updates(100, per_device_batch=8, grad_accum=4, num_epochs=3)
    assert p["micro_per_epoch"] == 12 and p["windows_per_epoch"] == 3 and p["total_opt_steps"] == 9
    assert p["dropped_samples_per_epoch"] == 4 and p["dropped_micro_per_epoch"] == 0
    p = tr.plan_updates(7, 8, 4, 1)
    assert p["total_opt_steps"] == 0


def test_iter_windows_drops_incomplete_tail():
    assert [len(w) for w in tr.iter_windows(range(10), 4)] == [4, 4]


@pytest.mark.parametrize("text,expected", [
    ("blah The answer is: 1,234.", 1234),
    ("The answer is: 18", 18),
    ("The answer is: 3/4", 1),
    ("The answer is: 5/0", 5),
    ("The answer is: -2.6", -3),
    ("no marker 42", None),
    ("The answer is: none", None),
    ("The answer is: 1 The answer is: 7", 7),
])
def test_official_extraction(text, expected):
    assert gs.extract_answer_number(text) == expected


def test_gold_and_scoring():
    assert gs.gold_value("reason #### 1,000") == 1000.0
    assert gs.gold_value("no marker") is None
    assert gs.gold_value("#### ") is None
    assert gs.score_primary("The answer is: 12", 12.0) == (True, 12)
    assert gs.score_primary("12", 12.0) == (False, None)  # 无标记：不做最后数字回退
    ok, _ = gs.score_secondary("The answer is: 2.5", "x #### 2.5")
    assert ok
    ok, _ = gs.score_secondary("The answer is: 3", "x #### 2.5")
    assert not ok


def test_apply_stop_takes_earliest():
    assert gs.apply_stop("abc Question: x Response", ["Response", "Question"]) == ("abc ", "Question")
    assert gs.apply_stop("abc", ["Question"]) == ("abc", None)


def test_metrics_definitions_and_zero_denominators():
    m = ev.compute_metrics(100, 90, 72, 54)
    assert m["accuracy"] == pytest.approx(0.6) and m["pred_parse_coverage"] == pytest.approx(0.8)
    assert m["accuracy_on_parsed"] == pytest.approx(0.75) and m["gold_valid_ratio"] == pytest.approx(0.9)
    z = ev.compute_metrics(10, 10, 0, 0)
    assert z["accuracy"] == 0.0 and z["accuracy_on_parsed"] is None


def test_pairing_contract():
    ev.check_pairing(2, [1, 2])
    with pytest.raises(RuntimeError):
        ev.check_pairing(3, [1, 2])


def test_stop_string_spec():
    assert ev.resolve_stop_strings(None) == gs.OFFICIAL_STOP_STRINGS_F7509EEE
    assert ev.resolve_stop_strings("none") == []
    with pytest.raises(ValueError):
        ev.resolve_stop_strings(5)


def test_bf16_update_lost_detection():
    from scripts.m1.diag_bf16_update import param_group, update_lost

    w = torch.ones(1000, dtype=torch.bfloat16)  # bf16 在 1.0 附近的间隔为 2^-7≈0.0078
    r = torch.ones(1000)
    lost, u_sum, applied = update_lost(w, r, 1e-3)  # 小于半间隔：全部丢失
    assert lost == 1000 and applied == 0.0 and u_sum == pytest.approx(1.0, rel=1e-5)
    lost, _, applied = update_lost(w, r, 1e-2)  # 大于半间隔：全部保留（舍入到一个间隔）
    assert lost == 0 and applied > 0
    lost, _, _ = update_lost(w, torch.zeros(1000), 1e-3)  # r=0 不计入“丢失”
    assert lost == 0
    assert param_group("model.layers.0.self_attn.q_proj.weight") == "q_proj"
    assert param_group("model.layers.0.input_layernorm.weight") == "norm"
    assert param_group("model.embed_tokens.weight") == "embed_tokens"
