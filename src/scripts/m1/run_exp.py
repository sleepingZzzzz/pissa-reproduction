"""M1 实验编排：分配全局实验号 n，串行运行各步骤，并按 n 管理 logs / ckpt / results。

一次启动 = 一个 n（取 logs 根目录最大 n + 1）。目录布局（创建时带 ``(fail)_``，成功后去掉）：
    logs/(fail)_exp{n}_m1_{模型名}[_{kind}]_{MMDD_HH}/      各步骤日志 + summary.txt
    checkpoints/{smoke,full}/(fail)_exp{n}_{MMDD_HH}/{method}/
    results/m1/{tokenize_stats,bf16_diag,smoke,full}/(fail)_exp{n}_{MMDD_HH}/metrics/

kind：
    smoke     少量样本训练 + 少量题评测；评测成功后自动删除该方法的 ckpt（失败则保留）；
              开始前先清空上一次 smoke 的 ckpt 与 results（只删 smoke 子目录内的版本目录）。
    full      正式训练 + 评测。
    tokenize  只做分词/截断统计（可传多个 --config）。
    bf16diag  BF16 + AdamW 小更新诊断。

fail 标识按类别判定：ckpt = 所有训练成功；results = 所有步骤成功；logs = 所有步骤成功。
全程不碰 git，不关机。

用法示例：
    python src/scripts/m1/run_exp.py --kind smoke --vllm-smoke
    python src/scripts/m1/run_exp.py --kind full --backend vllm
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

KINDS = ("smoke", "full", "tokenize", "bf16diag")
RESULT_SUBDIR = {"smoke": "smoke", "full": "full", "tokenize": "tokenize_stats", "bf16diag": "bf16_diag"}
TRAIN = "src/scripts/m1/train.py"
EVAL = "src/scripts/m1/eval_gsm8k.py"
DIAG = "src/scripts/m1/diag_bf16_update.py"


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
    args = p.parse_args()

    configs = args.config or ["configs/paper/m1_llama2_7b.yaml"]
    if len(configs) > 1 and args.kind != "tokenize":
        raise SystemExit("只有 --kind tokenize 支持多个 --config。")
    cfg = load_config(abspath(configs[0]))
    model_name = cfg["model"]["name"]
    tag = f"m1_{model_name}" + ("" if args.kind == "full" else f"_{args.kind}")

    logs_root = REPO_ROOT / "logs"
    n, st, logs_dir = alloc_logs_dir(logs_root, tag)
    runner = Runner(logs_dir)
    runner.note(f"exp{n}  kind={args.kind}  logs={logs_dir.name}")

    res_root = REPO_ROOT / "results" / "m1" / RESULT_SUBDIR[args.kind]
    has_ckpt = args.kind in ("smoke", "full")
    ckpt_root = REPO_ROOT / "checkpoints" / args.kind if has_ckpt else None
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

    if args.kind == "tokenize":
        for c in configs:
            runner.run(f"tokenize_{Path(c).stem}", [py, TRAIN, "--config", c, "--method", "pissa",
                                                    "--tokenize-only", "--overwrite", *paths])
    elif args.kind == "bf16diag":
        runner.run("bf16_diag", [py, DIAG, "--config", cfg_path, "--windows", str(args.windows), "--overwrite", *paths])
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
            ecmd = [py, EVAL, "--config", cfg_path, "--method", m, "--ckpt", str(mdir), *paths]
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
                        "--set", f"paths.output_dir={vres}"])
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
