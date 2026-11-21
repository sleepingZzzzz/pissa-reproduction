"""M1 实验编排：分配全局实验号 n，串行运行各步骤，并按 n 管理 logs / ckpt / results。

一次启动 = 一个 n（取 logs 根目录最大 n + 1）。目录布局（创建时带 ``(fail)_``，成功后去掉）：
    logs/(fail)_exp{n}_{阶段}_{模型名}[_{kind}]_{MMDD_HH}/  各步骤日志 + summary.txt
    阶段 m1：
        checkpoints/{smoke,full}/(fail)_exp{n}_{MMDD_HH}/{method}/
        results/m1/{tokenize_stats,bf16_diag,smoke,full}/(fail)_exp{n}_{MMDD_HH}/metrics/
    阶段 m2/m3（阶段取自配置 experiment 字段第一个下划线前的部分）：
        checkpoints/smoke/...（全局，每次 smoke 覆盖）；checkpoints/{阶段}/{模型名}/full/...
        results/{阶段}/{模型名}/{tokenize_stats,bf16_diag,smoke,full}/...（smoke 只清除同一模型下的旧版本）

kind：
    smoke     少量样本训练 + 少量题评测；评测成功后自动删除该方法的 ckpt（失败则保留）；
              开始前先清空上一次 smoke 的 ckpt 与 results（只删 smoke 子目录内的版本目录）。
    full      正式训练 + 评测。
    tokenize  只做分词/截断统计（可传多个 --config）。
    bf16diag  BF16 + AdamW 小更新诊断。
    base      不训练，直接评测原始基座（配置 model.local_path 或 --base-model-dir），无 ckpt；
              用 --fallback-score 可同时输出“取最后一个数字”的回退评分（辅助指标）。
    table4    M3 Table 4 量化重建误差（不训练、无 ckpt）：对 --rounds 中的每个轮数串行运行
              src/scripts/m3/table4.py，结果写入 results/{阶段}/{模型名}/table4/exp{n}_{MMDD_HH}/metrics/。

fail 标识按类别判定：ckpt = 所有训练成功；results = 所有步骤成功；logs = 所有步骤成功。
全程不碰 git，不关机。

用法示例：
    python src/scripts/m1/run_exp.py --kind smoke --vllm-smoke
    python src/scripts/m1/run_exp.py --kind full --backend vllm
    python src/scripts/m1/run_exp.py --kind base --config configs/paper/m2_qwen25_7b.yaml --backend vllm --fallback-score
    python src/scripts/m1/run_exp.py --kind table4 --config configs/paper/m3_table4_llama3_8b.yaml --rounds 5 1
    python src/scripts/m1/run_exp.py --kind tokenize --config configs/paper/m1_llama2_7b.yaml --config configs/paper/m1_llama2_7b_len1024.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import shutil
import subprocess
import time

from exp_registry import alloc_logs_dir, category_dir, clear_versions, mark_ok
from utils import REPO_ROOT, abspath, load_config

KINDS = ("smoke", "full", "tokenize", "bf16diag", "base", "table4")
RESULT_SUBDIR = {"smoke": "smoke", "full": "full", "tokenize": "tokenize_stats", "bf16diag": "bf16_diag",
                 "base": "base", "table4": "table4"}
TRAIN = "src/scripts/m1/train.py"
EVAL = "src/scripts/m1/eval_gsm8k.py"
DIAG = "src/scripts/m1/diag_bf16_update.py"
TABLE4 = "src/scripts/m3/table4.py"


class Runner:
    """顺序执行子进程，日志写入 logs 目录，并把每步返回码追加到 summary.txt。"""

    def __init__(self, logs_dir: Path):
        self.logs_dir = logs_dir
        self.rcs: dict[str, int] = {}

    def run(self, name: str, cmd: list[str]) -> int:
        t0 = time.time()
        self._note(f"START {name}")
        with open(self.logs_dir / f"{name}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        self.rcs[name] = rc
        self._note(f"END   {name} rc={rc} ({time.time() - t0:.0f}s)")
        return rc

    def note(self, msg: str) -> None:
        self._note(msg)

    def _note(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.logs_dir / "summary.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @property
    def all_ok(self) -> bool:
        return all(rc == 0 for rc in self.rcs.values())


def check_learned_format(runner: Runner, eval_json: Path, method: str) -> int:
    """读取 smoke 评测结果：N_pred>0 且 has_eos_count>0 才算模型学到了回答格式与结束符。"""
    try:
        r = json.loads(eval_json.read_text(encoding="utf-8"))["results"]
        n_pred, n_eos = int(r["N_pred"]), int(r["has_eos_count"])
    except (OSError, KeyError, ValueError, TypeError) as e:
        runner.note(f"check_format {method}: 无法读取 {eval_json}（{e!r}），按失败处理")
        runner.rcs[f"check_format_{method}"] = 1
        return 1
    ok = n_pred > 0 and n_eos > 0
    runner.note(f"check_format {method}: N_pred={n_pred} has_eos_count={n_eos} -> {'OK' if ok else 'FAIL'}")
    runner.rcs[f"check_format_{method}"] = 0 if ok else 1
    return 0 if ok else 1


STAGES = ("m1", "m2", "m3")


def stage_layout(cfg: dict, kind: str, repo_root: Path) -> tuple[str, str, Path, Path | None]:
    """按配置的阶段（``experiment`` 第一个下划线前的部分）与模型名决定标签与产物根目录。

    Args:
        cfg: 含 ``experiment`` 与 ``model.name`` 的配置字典。
        kind: ``KINDS`` 之一。
        repo_root: 仓库根目录。

    Returns:
        ``(阶段, logs 名中的标签, results 根目录, ckpt 根目录或 None)``。
        m1 保持原布局；其他阶段的 results 放在 ``results/{阶段}/{模型名}/{类别}``，
        full 的 ckpt 放在 ``checkpoints/{阶段}/{模型名}/full``；
        smoke 的 ckpt 仍在全局 ``checkpoints/smoke``（每次 smoke 覆盖）。

    Raises:
        SystemExit: ``experiment`` 存在但阶段不是 m1/m2/m3；缺省该字段时按 m1 处理。
    """
    exp_name = cfg.get("experiment")
    # 配置没有 experiment 字段时沿用 m1 布局（向后兼容）；有但无法识别则报错，不猜。
    stage = "m1" if exp_name is None else str(exp_name).split("_")[0]
    if stage not in STAGES:
        raise SystemExit(f"无法从 experiment={cfg.get('experiment')!r} 判断阶段（应以 m1_/m2_/m3_ 开头）。")
    model_name = cfg["model"]["name"]
    tag = f"{stage}_{model_name}" + ("" if kind == "full" else f"_{kind}")
    sub = RESULT_SUBDIR[kind]
    if stage == "m1":
        res_root = repo_root / "results" / "m1" / sub
        full_ckpt = repo_root / "checkpoints" / "full"
    else:
        res_root = repo_root / "results" / stage / model_name / sub
        full_ckpt = repo_root / "checkpoints" / stage / model_name / "full"
    if kind == "smoke":
        ckpt_root: Path | None = repo_root / "checkpoints" / "smoke"
    elif kind == "full":
        ckpt_root = full_ckpt
    else:
        ckpt_root = None
    return stage, tag, res_root, ckpt_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--config", action="append", default=None,
                   help="默认 configs/paper/m1_llama2_7b.yaml；tokenize 可重复传入多个")
    p.add_argument("--methods", nargs="+", default=["pissa", "lora", "full"], choices=["pissa", "lora", "full"])
    p.add_argument("--limit", type=int, default=256, help="smoke 训练样本数（256 = 2 个更新步）")
    p.add_argument("--eval-limit", type=int, default=20, help="smoke 评测题数")
    p.add_argument("--backend", default="hf", choices=["hf", "vllm"], help="full 的评测后端；vllm 失败回退 hf")
    p.add_argument("--require-pred", action="store_true",
                   help="smoke：评测结果 N_pred=0 或 has_eos_count=0（模型没学会格式/结束符）视为失败")
    p.add_argument("--vllm-smoke", action="store_true", help="smoke：对 --vllm-method 额外做一次 vLLM 评测")
    p.add_argument("--vllm-method", default="pissa")
    p.add_argument("--vllm-limit", type=int, default=50)
    p.add_argument("--vllm-python", default="/root/autodl-tmp/venv-vllm2/bin/python")
    p.add_argument("--master-weights", default="cpu_fp32", choices=["none", "cpu_fp32"],
                   help="仅用于 method=full 的 train.py --master-weights")
    p.add_argument("--windows", type=int, default=4, help="bf16diag 的梯度窗口数")
    p.add_argument("--fallback-score", action="store_true",
                   help="评测时额外计算回退评分 last_number_v1（辅助指标，不替代主评分）；默认关闭")
    p.add_argument("--rounds", type=int, nargs="+", default=[5, 1],
                   help="仅 --kind table4：QPiSSA 外层交替轮数，按给定顺序串行运行（默认 5 1）")
    p.add_argument("--base-model-dir", default=None,
                   help="仅 --kind base：基座模型目录；缺省取配置 model.local_path")
    args = p.parse_args()

    configs = args.config or ["configs/paper/m1_llama2_7b.yaml"]
    if len(configs) > 1 and args.kind != "tokenize":
        raise SystemExit("只有 --kind tokenize 支持多个 --config。")
    cfg = load_config(abspath(configs[0]))
    stage, tag, res_root, ckpt_root = stage_layout(cfg, args.kind, REPO_ROOT)

    logs_root = REPO_ROOT / "logs"
    n, st, logs_dir = alloc_logs_dir(logs_root, tag)
    runner = Runner(logs_dir)
    runner.note(f"exp{n}  kind={args.kind}  logs={logs_dir.name}")

    has_ckpt = ckpt_root is not None
    if args.kind == "smoke":
        for root in (res_root, ckpt_root):
            removed = clear_versions(root)
            runner.note(f"smoke 覆盖：清除 {root} 下的旧版本 {removed}")
    res_dir = category_dir(res_root, n, st)
    res_dir.mkdir(parents=True)
    ckpt_dir = category_dir(ckpt_root, n, st) if has_ckpt else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True)

    py = sys.executable
    paths = ["--set", f"paths.output_dir={res_dir}"]
    if ckpt_dir is not None:
        paths += ["--set", f"paths.ckpt_dir={ckpt_dir}"]
    cfg_path = configs[0]
    train_ok = True
    fb = ["--fallback-score"] if args.fallback_score else []

    if args.kind == "tokenize":
        for c in configs:
            runner.run(f"tokenize_{Path(c).stem}", [py, TRAIN, "--config", c, "--method", "pissa",
                                                    "--tokenize-only", "--overwrite", *paths])
    elif args.kind == "bf16diag":
        runner.run("bf16_diag", [py, DIAG, "--config", cfg_path, "--windows", str(args.windows), "--overwrite", *paths])
    elif args.kind == "table4":
        for r in args.rounds:
            runner.run(f"table4_rounds{r}", [py, TABLE4, "--config", cfg_path, "--rounds", str(r),
                                             "--set", f"paths.output_dir={res_dir / 'metrics'}"])
    elif args.kind == "base":
        base_dir = args.base_model_dir or cfg["model"].get("local_path")
        if not base_dir:
            runner.note("base 评测需要 --base-model-dir 或配置 model.local_path")
            runner.rcs["base_model_dir"] = 1
        else:
            # 基座没有适配器：method=full 只表示“按普通模型加载”，不会注入或合并任何东西。
            bcmd = [py, EVAL, "--config", cfg_path, "--method", "full", "--ckpt", str(abspath(base_dir)),
                    "--overwrite", *paths, *fb]
            if args.backend == "vllm":
                if runner.run("eval_base", [args.vllm_python, *bcmd[1:], "--backend", "vllm"]) != 0:
                    runner.run("eval_base_hf", [*bcmd, "--backend", "hf"])
            else:
                runner.run("eval_base", [*bcmd, "--backend", "hf"])
    else:
        smoke = args.kind == "smoke"
        for m in args.methods:
            tcmd = [py, TRAIN, "--config", cfg_path, "--method", m, *paths]
            if smoke:
                tcmd += ["--limit", str(args.limit), "--overwrite"]
            if m == "full" and args.master_weights != "none":
                tcmd += ["--master-weights", args.master_weights]
            if runner.run(f"train_{m}", tcmd) != 0:
                train_ok = False
                continue
            mdir = ckpt_dir / m
            ecmd = [py, EVAL, "--config", cfg_path, "--method", m, "--ckpt", str(mdir), *paths, *fb]
            if smoke:
                ecmd += ["--limit", str(args.eval_limit), "--backend", "hf", "--overwrite"]
                rc = runner.run(f"eval_{m}", ecmd)
                if rc == 0 and args.require_pred:
                    rc = check_learned_format(runner, res_dir / "metrics" / f"eval_{m}.json", m)
                if args.vllm_smoke and m == args.vllm_method:
                    vres = res_dir / "vllm"
                    rc |= runner.run(f"eval_{m}_vllm", [
                        args.vllm_python, EVAL, "--config", cfg_path, "--method", m, "--ckpt", str(mdir),
                        "--limit", str(args.vllm_limit), "--backend", "vllm", "--overwrite",
                        "--set", f"paths.output_dir={vres}", *fb])
                if rc == 0:
                    shutil.rmtree(mdir)
                    runner.note(f"smoke 评测成功，已删除 ckpt：{mdir}")
                else:
                    runner.note(f"smoke 评测失败，保留 ckpt：{mdir}")
            elif args.backend == "vllm":
                if runner.run(f"eval_{m}", [args.vllm_python, *ecmd[1:], "--backend", "vllm"]) != 0:
                    runner.run(f"eval_{m}_hf", [*ecmd, "--backend", "hf", "--overwrite"])
            else:
                runner.run(f"eval_{m}", [*ecmd, "--backend", "hf"])

    # ---- 收尾：按类别去掉 fail 前缀 ----
    ok_all = runner.all_ok
    if args.kind == "base" and args.backend == "vllm" and "base_model_dir" not in runner.rcs:
        ok_all = runner.rcs.get("eval_base") == 0 or runner.rcs.get("eval_base_hf") == 0
    if args.kind == "full" and args.backend == "vllm":
        # vLLM 失败但 hf 回退成功时，回退产物完整：以“每个方法最终有一次成功评测”为准。
        final_ok = all(
            runner.rcs.get(f"train_{m}") == 0
            and (runner.rcs.get(f"eval_{m}") == 0 or runner.rcs.get(f"eval_{m}_hf") == 0)
            for m in args.methods)
        ok_all = final_ok
    if ckpt_dir is not None:
        if not any(ckpt_dir.iterdir()):
            ckpt_dir.rmdir()
            runner.note(f"ckpt 目录已空，移除：{ckpt_dir.name}")
        elif train_ok:
            ckpt_dir = mark_ok(ckpt_dir)
    if ok_all:
        res_dir = mark_ok(res_dir)
    runner.note(f"results={res_dir.name}" + ("" if ok_all else "  (有失败步骤，保留 fail 标识)"))
    logs_final = logs_dir
    if ok_all:
        runner.note("ALL OK")
        logs_final = mark_ok(logs_dir)
    else:
        runner.note("存在失败步骤，logs 保留 fail 标识")
    print(f"logs: {logs_final}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
