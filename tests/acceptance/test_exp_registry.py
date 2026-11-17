"""实验编号/命名与 run_exp 编排验收（新增文件；全部在 tmp_path 内，不触碰真实 logs/results/checkpoints）。"""
import sys
from datetime import datetime

import pytest
import yaml

from exp_registry import (alloc_logs_dir, category_dir, clear_versions, find_by_n, mark_ok, max_n,
                          parse_name)


def test_parse_name():
    assert parse_name("exp12_m1_x_1116_14") == (12, False)
    assert parse_name("(fail)_exp3_1116_14") == (3, True)
    assert parse_name("m1_1116_0110") is None
    assert parse_name("exp_abc") is None
    assert parse_name("xexp5_a") is None


def test_max_n_ignores_other_entries_and_counts_files_and_fail(tmp_path):
    (tmp_path / "m1_old").mkdir()
    (tmp_path / "old.log").write_text("x")
    (tmp_path / "exp2_a_0101_01").mkdir()
    (tmp_path / "(fail)_exp7_b_0101_02").mkdir()
    (tmp_path / "exp3_c.log").write_text("x")
    assert max_n(tmp_path) == 7  # 不是条目个数
    assert max_n(tmp_path / "nonexistent") == 0


def test_alloc_logs_dir_increments_and_names(tmp_path):
    now = datetime(2026, 11, 16, 14, 5)
    n1, st, p1 = alloc_logs_dir(tmp_path, "m1_Llama-2-7b_smoke", now)
    n2, _, p2 = alloc_logs_dir(tmp_path, "m1_Llama-2-7b", now)
    assert (n1, n2) == (1, 2) and st == "1116_14"
    assert p1.name == "(fail)_exp1_m1_Llama-2-7b_smoke_1116_14"
    assert p2.is_dir()


def test_find_by_n_both_forms_and_mark_ok(tmp_path):
    d = category_dir(tmp_path, 4, "1116_14")
    d.mkdir()
    assert find_by_n(tmp_path, 4) == d
    assert find_by_n(tmp_path, 5) is None
    ok = mark_ok(d)
    assert ok.name == "exp4_1116_14" and ok.exists() and not d.exists()
    assert find_by_n(tmp_path, 4) == ok
    assert mark_ok(ok) == ok
    category_dir(tmp_path, 4, "1117_01", fail=False).mkdir()
    with pytest.raises(RuntimeError):
        find_by_n(tmp_path, 4)


def test_clear_versions_only_removes_versioned_dirs(tmp_path):
    (tmp_path / "exp1_a").mkdir()
    (tmp_path / "(fail)_exp2_a").mkdir()
    (tmp_path / "keep").mkdir()
    (tmp_path / "exp3_file").write_text("x")
    assert clear_versions(tmp_path) == ["(fail)_exp2_a", "exp1_a"]
    assert (tmp_path / "keep").exists() and (tmp_path / "exp3_file").exists()


STUB = """import sys, pathlib
a = sys.argv[1:]
def val(k): return a[a.index(k) + 1] if k in a else None
paths = {}
for i, x in enumerate(a):
    if x == "--set":
        k, v = a[i + 1].split("=", 1); paths[k] = v
method = val("--method")
fail = pathlib.Path(__file__).with_name("fail_" + str(method))
out = pathlib.Path(paths["paths.output_dir"]) / "metrics"; out.mkdir(parents=True, exist_ok=True)
if "--tokenize-only" in a:
    (out / "tok.json").write_text("{}")
elif "train" in pathlib.Path(__file__).name:
    c = pathlib.Path(paths["paths.ckpt_dir"]) / method; c.mkdir(parents=True, exist_ok=True)
    (c / "model.safetensors").write_text("w")
    (out / f"train_{method}.json").write_text("{}")
else:
    assert (pathlib.Path(val("--ckpt")) / "model.safetensors").exists()
    bad = pathlib.Path(__file__).with_name("zero_pred").exists()
    (out / f"eval_{method}.json").write_text(
        '{"results": {"N_pred": %d, "has_eos_count": %d}}' % (0 if bad else 3, 0 if bad else 2))
sys.exit(1 if fail.exists() and "eval" in pathlib.Path(__file__).name else 0)
"""


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    import scripts.m1.run_exp as re_
    (tmp_path / "logs" / "exp5_old.log").parent.mkdir(parents=True)
    (tmp_path / "logs" / "exp5_old.log").write_text("x")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name in ("train.py", "eval.py", "diag.py"):
        (stubs / name).write_text(STUB)
    monkeypatch.setattr(re_, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(re_, "TRAIN", str(stubs / "train.py"))
    monkeypatch.setattr(re_, "EVAL", str(stubs / "eval.py"))
    monkeypatch.setattr(re_, "DIAG", str(stubs / "diag.py"))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"model": {"name": "Toy"}}))
    return re_, tmp_path, stubs, str(cfg)


def _run(re_, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_exp.py", *argv])
    with pytest.raises(SystemExit) as e:
        re_.main()
    return e.value.code


def test_run_exp_smoke_success_deletes_ckpt_and_clears_fail(fake_repo, monkeypatch):
    re_, root, stubs, cfg = fake_repo
    assert _run(re_, monkeypatch, ["--kind", "smoke", "--config", cfg]) == 0
    logs = sorted(p.name for p in (root / "logs").iterdir() if p.is_dir())
    assert len(logs) == 1 and logs[0].startswith("exp6_m1_Toy_smoke_")  # 接在 exp5 之后，且无 fail
    res = list((root / "results/m1/smoke").iterdir())
    assert len(res) == 1 and res[0].name.startswith("exp6_")
    assert (res[0] / "metrics/eval_pissa.json").exists()
    assert list((root / "checkpoints/smoke").iterdir()) == []  # 成功后 ckpt 全删，空目录也移除


def test_run_exp_smoke_failure_keeps_ckpt_and_fail_and_next_smoke_overwrites(fake_repo, monkeypatch):
    re_, root, stubs, cfg = fake_repo
    (stubs / "fail_lora").write_text("x")
    assert _run(re_, monkeypatch, ["--kind", "smoke", "--config", cfg]) == 1
    logs = [p.name for p in (root / "logs").iterdir() if p.is_dir()]
    assert logs[0].startswith("(fail)_exp6_")
    ck = list((root / "checkpoints/smoke").iterdir())
    assert len(ck) == 1 and ck[0].name.startswith("exp6_")  # 训练全部成功 -> ckpt 类别无 fail
    assert [p.name for p in ck[0].iterdir()] == ["lora"]  # 仅失败方法的 ckpt 保留
    assert list((root / "results/m1/smoke").iterdir())[0].name.startswith("(fail)_exp6_")
    (stubs / "fail_lora").unlink()
    assert _run(re_, monkeypatch, ["--kind", "smoke", "--config", cfg]) == 0
    assert [p.name.split("_")[0] for p in (root / "results/m1/smoke").iterdir()] == ["exp7"]  # 旧版本被覆盖
    assert list((root / "checkpoints/smoke").iterdir()) == []


def test_run_exp_full_keeps_versions(fake_repo, monkeypatch):
    re_, root, stubs, cfg = fake_repo
    assert _run(re_, monkeypatch, ["--kind", "full", "--config", cfg, "--methods", "pissa"]) == 0
    assert _run(re_, monkeypatch, ["--kind", "full", "--config", cfg, "--methods", "pissa"]) == 0
    names = sorted(p.name.split("_")[0] for p in (root / "results/m1/full").iterdir())
    assert names == ["exp6", "exp7"]
    assert (list((root / "checkpoints/full").iterdir())[0] / "pissa/model.safetensors").exists()


def test_run_exp_tokenize_multi_config(fake_repo, monkeypatch):
    re_, root, stubs, cfg = fake_repo
    assert _run(re_, monkeypatch, ["--kind", "tokenize", "--config", cfg, "--config", cfg]) == 0
    assert len(list((root / "results/m1/tokenize_stats").iterdir())) == 1


def test_require_pred_gates_smoke_cleanup(fake_repo, monkeypatch):
    re_, root, stubs, cfg = fake_repo
    (stubs / "zero_pred").write_text("x")
    assert _run(re_, monkeypatch, ["--kind", "smoke", "--config", cfg, "--methods", "pissa", "--require-pred"]) == 1
    ck = list((root / "checkpoints/smoke").iterdir())
    assert (ck[0] / "pissa").exists()  # N_pred=0 -> 保留 ckpt
    assert list((root / "logs").glob("(fail)_exp6_*"))
    (stubs / "zero_pred").unlink()
    assert _run(re_, monkeypatch, ["--kind", "smoke", "--config", cfg, "--methods", "pissa", "--require-pred"]) == 0
    assert list((root / "checkpoints/smoke").iterdir()) == []
