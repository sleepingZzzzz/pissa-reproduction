"""M0：分别验证重构、主成分选择、Fast SVD 质量及受控 PEFT 对齐。

重构只验证接线；解析谱样本验证选择。随机近似质量和参考实现一致性是不同结论。
niter=0 敏感性行仅为诊断，不是新的验收标准。既有良好设置曾观察到约 1.034 的
质量比，当前保留 1.50 安全界限，不能称为盲校准。PEFT 检查包含用户指定秩和
固定的 r/k=16/256 区间。小型报告与命令行工具沿用既有 M0 组织。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from pissa import PiSSALinear, pissa_init  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "results/m0/exp1"
# Predeclared normwise tolerances: FP32 implementation vs FP64 oracle.
# Absolute allowance scales with ||W||, including at a zero optimal tail.
EXACT_RTOL, ROUNDING_RTOL = 3e-4, 3e-5
# Broad approximation safety bounds, not universal near-optimality claims.
FAST_LIMITS = {"gaussian": 1.50, "decaying": 1.50, "gapped": 1.50}


class ArgumentParser64(argparse.ArgumentParser):
    """将命令行参数错误的退出码统一为 64 的 ArgumentParser。"""
    def error(self, message):
        """打印 usage 和错误消息，以退出码 64 结束进程。"""
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


def atomic_json(path, payload):
    """以 UTF-8 临时文件和原子替换写入 JSON，拒绝 NaN，并在结束时清理临时文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def environment(device="cpu"):
    """采集解释器、依赖版本、设备及 Git 状态；命令参数中的疑似密钥做脱敏。"""
    versions = {}
    for name in ("torch", "torchvision", "peft", "transformers", "safetensors", "matplotlib", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git = {"commit": None, "dirty": None, "reason": None}
    try:
        git["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        git["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
            text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError) as exc:
        git["reason"] = f"{type(exc).__name__}: {exc}"
    command, hide = [], False
    for arg in sys.argv:
        secret = any(key in arg.lower() for key in ("token", "password", "secret", "api-key"))
        command.append("<redacted>" if hide or secret else arg)
        hide = secret and "=" not in arg
    return {"python": sys.version, "executable": sys.executable, "versions": versions,
        "cuda_build": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "torch_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "initialization_dtype": "torch.float32", "command": command, "git": git}


def new_report(experiment, config, device="cpu"):
    """创建含配置、环境和 running 状态的 schema_version=2 报告字典。"""
    return {"schema_version": 2, "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(), "status": "running", "exit_code": None,
        "config": config, "environment": environment(device), "results": {}, "timings_seconds": {}}


def finish_report(payload, path, status, code, started):
    """更新最终状态、退出码和总耗时，原子写报告并返回传入的退出码。"""
    payload.update(status=status, exit_code=code)
    payload["timings_seconds"]["total"] = time.perf_counter() - started
    atomic_json(path, payload)
    return code


def run_checks(path, config, checks, device="cpu"):
    """逐项执行检查并持续写报告，区分通过、失败、异常与跳过。

    Args:
        path: 报告 JSON 路径。
        config: 写入报告的配置。
        checks: 检查名称到无参数可调用对象的有序映射。
        device: 用于环境记录的设备。

    Returns:
        全部通过为 0，失败或异常为 1，包含跳过且无失败为 2。
    """
    started = time.perf_counter()
    payload = new_report("M0/verify_init", config, device)
    payload["results"] = {n: {"status": "pending", "executed": False} for n in checks}
    atomic_json(path, payload)
    for name, check in checks.items():
        payload["results"][name] = {"status": "running", "executed": True}
        atomic_json(path, payload)
        tick = time.perf_counter()
        try:
            result = check()
            state = "skipped" if result.get("skipped") else "passed" if result.get("passed") else "failed"
            result.update(status=state, executed=state != "skipped", skipped=state == "skipped")
        except Exception as exc:
            result = {"status": "error", "executed": True, "skipped": False,
                "exception_type": type(exc).__name__, "reason": str(exc)}
        payload["results"][name] = result
        payload["timings_seconds"][name] = time.perf_counter() - tick
        print(f"{name}: {result['status']} ({payload['timings_seconds'][name]:.3f}s)", flush=True)
        atomic_json(path, payload)
    states = [x["status"] for x in payload["results"].values()]
    executed = [s for s in states if s != "skipped"]
    payload["executed_all_pass"] = bool(executed) and all(s == "passed" for s in executed)
    payload["all_pass"] = bool(states) and all(s == "passed" for s in states)
    status, code = ("error", 1) if "error" in states else ("failed", 1) if "failed" in states else (
        ("partial", 2) if "skipped" in states else ("passed", 0))
    return finish_report(payload, path, status, code, started)


@contextmanager
def seeded(seed, device="cpu"):
    """临时设置指定 CPU/CUDA 随机流，退出上下文后恢复原状态。

    Args:
        seed: 上下文内的随机种子。
        device: CPU 或要隔离随机流的 CUDA 设备。

    Yields:
        无返回值的随机流隔离上下文。
    """
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
        yield


def norm(x):
    """将张量脱离梯度并转为 float64，返回展平后的欧氏范数。"""
    return float(torch.linalg.vector_norm(x.detach().double()))


def relative(a, b):
    """返回 ||a-b||/||b||；参考范数为零时返回 None。"""
    denominator = norm(b)
    return norm(a.detach().double()-b.detach().double()) / denominator if denominator else None


def close(a, b, scale=None, rtol=ROUNDING_RTOL):
    """按有限值和混合绝对/相对范数误差判断接近，不执行逐元素 allclose。"""
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return False
    scale = norm(b) if scale is None else float(scale)
    return norm(a.detach().double()-b.detach().double()) <= rtol*norm(b) + ROUNDING_RTOL*max(scale, 1e-30)


def known_spectrum(out_f, in_f, r, kind="gapped", seed=0):
    """用已知奇异谱构造测试矩阵，返回 FP32 权重、FP64 主成分及完整谱。"""
    k = min(out_f, in_f)
    with seeded(seed):
        u = torch.linalg.qr(torch.randn(out_f, k, dtype=torch.float64), mode="reduced").Q
        v = torch.linalg.qr(torch.randn(in_f, k, dtype=torch.float64), mode="reduced").Q
    if kind == "gapped":
        s = torch.cat((torch.linspace(10., 5., r, dtype=torch.float64),
                       torch.linspace(.5, .05, k-r, dtype=torch.float64)))
    elif kind == "decaying":
        s = torch.logspace(1., -2., k, dtype=torch.float64)
    elif kind == "repeated":
        s = torch.ones(k, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown spectrum: {kind}")
    return ((u*s) @ v.T).float(), (u[:, :r]*s[:r]) @ v[:, :r].T, s


def assess_decomposition(w, delta, residual, r, *, expected_s=None, expected_delta=None):
    """核对有限值、重构、残差范数和秩；谱隙稳定时才比较主成分方向。"""
    if not all(bool(torch.isfinite(t).all()) for t in (w, delta, residual)):
        return {"passed": False, "reason": "non-finite decomposition"}
    wd, dd, rd = [t.detach().double() for t in (w, delta, residual)]
    s = torch.linalg.svdvals(wd) if expected_s is None else expected_s.double().to(w.device)
    expected_fro = float(torch.linalg.vector_norm(s[r:]))
    expected_spec = float(s[r]) if r < len(s) else 0.
    actual_fro, actual_spec = norm(rd), float(torch.linalg.matrix_norm(rd, ord=2))
    scale = norm(wd)
    atol = ROUNDING_RTOL*max(scale, 1e-30)
    gap = float(s[r-1]-s[r]) if r < len(s) else None
    directional = gap is None or gap > 1e-3*max(float(s[0]), 1e-30)
    checks = {"reconstruction": close(rd+dd, wd, scale),
        "residual_frobenius": abs(actual_fro-expected_fro) <= atol+EXACT_RTOL*expected_fro,
        "residual_spectral": abs(actual_spec-expected_spec) <= atol+EXACT_RTOL*expected_spec,
        "rank_at_most_r": bool(torch.linalg.svdvals(dd)[r:].max() <= atol) if r < min(w.shape) else True}
    if expected_delta is not None and directional:
        checks["known_principal"] = close(dd, expected_delta.to(w.device), scale, EXACT_RTOL)
    return {"passed": all(checks.values()), "checks": checks,
        "frobenius_actual": actual_fro, "frobenius_expected": expected_fro,
        "spectral_actual": actual_spec, "spectral_expected": expected_spec,
        "principal_rel_err": relative(dd, expected_delta.to(w.device)) if expected_delta is not None else None,
        "atol": atol, "rtol": EXACT_RTOL, "gap": gap, "direction_comparison_stable": directional,
        "oracle": "analytic_spectrum" if expected_s is not None else "independent_fp64_svdvals"}


def check_reconstruction(in_f, out_f, r, niter=None, device="cpu"):
    """检查合并权重与前向保持初始映射；该检查不能证明选中了主成分。"""
    with seeded(0, device):
        layer = nn.Linear(in_f, out_f).to(device)
        adapted = PiSSALinear.from_linear(layer, r, alpha=r, niter=niter)
        x = torch.randn(8, in_f, device=device)
    with torch.no_grad():
        reference, actual = layer(x), adapted(x)
    return {"passed": close(adapted.merged_weight(), layer.weight) and close(actual, reference),
        "reconstruction_rel_err": relative(adapted.merged_weight(), layer.weight),
        "forward_rel_err": relative(actual, reference), "rtol": ROUNDING_RTOL,
        "claim": "arithmetic/function preservation, not top-r selection"}


def check_svd_property(in_f, out_f, r, device="cpu"):
    """使用已知谱与独立 FP64 参考核对精确 SVD 的主成分和最优残差性质。"""
    samples = []
    for kind in ("gapped", "decaying", "repeated", "gaussian"):
        if kind == "gaussian":
            with seeded(0, device):
                w = torch.randn(out_f, in_f, device=device)
            expected, s = None, None
        else:
            w, expected, s = known_spectrum(out_f, in_f, r, kind)
            w = w.to(device)
        init = pissa_init(w, r)
        result = assess_decomposition(w, init.lora_B @ init.lora_A, init.residual, r,
                                     expected_s=s, expected_delta=expected)
        samples.append({"spectrum": kind, **result})
    return {"passed": all(x["passed"] for x in samples), "samples": samples}


def principal_angles(a, b):
    """对两组列空间做 QR/SVD，返回主夹角的角度值列表。"""
    qa = torch.linalg.qr(a.double(), mode="reduced").Q
    qb = torch.linalg.qr(b.double(), mode="reduced").Q
    cosine = torch.linalg.svdvals(qa.T @ qb).clamp(0, 1)
    return torch.rad2deg(torch.acos(cosine)).cpu().tolist()


def check_fast_quality(in_f, out_f, r, niters, seeds, device="cpu"):
    """按固定矩阵、种子、q 和迭代次数检查 Fast SVD 质量，返回逐样本报告。"""
    rows = []
    for kind in FAST_LIMITS:
        if kind == "gaussian":
            with seeded(21, device):
                w = torch.randn(out_f, in_f, device=device)
        else:
            w = known_spectrum(out_f, in_f, r, kind, 21)[0].to(device)
        u, s, vh = torch.linalg.svd(w.double(), full_matrices=False)
        optimum, scale = float(torch.linalg.vector_norm(s[r:])), norm(w)
        for niter in niters:
            for q in sorted({r, min(min(w.shape), r+8)}):
                for seed in seeds:
                    with seeded(seed, device):
                        tick = time.perf_counter()
                        init = pissa_init(w, r, niter=niter, q=q)
                        if str(device).startswith("cuda"):
                            torch.cuda.synchronize(device)
                        elapsed = time.perf_counter()-tick
                    delta = init.lora_B.float() @ init.lora_A.float()
                    residual = (w-delta).double()
                    actual = norm(residual)
                    near_zero = optimum <= ROUNDING_RTOL*max(scale, 1e-30)
                    ratio = None if near_zero else actual/optimum
                    passed = close(init.residual, w-delta, scale) and close(init.residual+delta, w, scale)
                    passed = passed and bool(torch.isfinite(residual).all()) and (
                        actual <= ROUNDING_RTOL*max(scale, 1e-30) if near_zero else ratio <= FAST_LIMITS[kind]+ROUNDING_RTOL)
                    gap = float(s[r-1]-s[r]) if r < len(s) else None
                    stable = gap is None or gap > 1e-3*max(float(s[0]), 1e-30)
                    rows.append({"spectrum": kind, "r": r, "q": q, "niter": niter, "seed": seed,
                        "residual_frobenius": actual, "optimal_frobenius": optimum,
                        "quality_ratio": ratio, "quality_limit": FAST_LIMITS[kind],
                        "residual_spectral": float(torch.linalg.matrix_norm(residual, ord=2)),
                        "optimal_spectral": float(s[r]) if r < len(s) else 0.,
                        "input_angles_degrees": principal_angles(init.lora_A.T, vh[:r].T) if stable else None,
                        "output_angles_degrees": principal_angles(init.lora_B, u[:, :r]) if stable else None,
                        "direction_comparison_stable": stable, "passed": bool(passed), "seconds": elapsed})
    groups = []
    for kind in FAST_LIMITS:
        for niter in niters:
            for q in sorted({row["q"] for row in rows}):
                group = [x for x in rows if (x["spectrum"], x["niter"], x["q"]) == (kind, niter, q)]
                values = [x["quality_ratio"] for x in group if x["quality_ratio"] is not None]
                groups.append({"spectrum": kind, "niter": niter, "q": q, "n": len(group),
                    "mean": statistics.mean(values) if values else None,
                    "sample_std": statistics.stdev(values) if len(values)>1 else None,
                    "worst": max(values) if values else None,
                    "pass_rate": sum(x["passed"] for x in group)/len(group)})
    return {"passed": all(x["passed"] for x in rows), "samples": rows, "summary": groups,
        "interpretation": "predeclared safety bounds; report actual quality, not universal near-optimality"}


def check_fast_sensitivity(in_f, out_f, r, seeds, device="cpu"):
    """对相同矩阵和种子运行预定的无幂迭代诊断。

    q 不得小于 r，q=r 是较小的合法求解维度。niter=0 用于探测质量敏感性，
    不保证每个样本都会变差；超过既有质量阈值在这里仅记录，不判定验收失败。
    """
    result = check_fast_quality(in_f, out_f, r, [0], seeds, device)
    rows = []
    for sample in result["samples"]:
        row = dict(sample)
        row["within_existing_limit"] = row.pop("passed")
        rows.append(row)
    return {"passed": all(torch.isfinite(torch.tensor(x["residual_frobenius"])) for x in rows),
        "diagnostic_only": True, "quality_outcomes_affect_all_pass": False,
        "retention_informed_by_prior_results": True,
        "threshold_policy": "retain 1.50 unchanged; prior good-run worst ratio about 1.034",
        "calibration_disclosure": "retention decision is informed by already observed good settings; not blind validation",
        "settings": {"niters": [0], "q": sorted({r, min(min(in_f, out_f), r+8)}),
                     "matrix_seed": 21, "seeds": list(seeds)},
        "samples": rows,
        "interpretation": "report every probe; niter=0 need not cross 1.50; no thresholds or earlier criteria changed"}


def import_peft():
    """按需导入 PEFT 接口；不可用时返回空接口和明确的跳过原因。"""
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
        import peft.tuners.lora.layer as lora_layer
        return (LoraConfig, get_peft_model, LlamaConfig, LlamaForCausalLM, lora_layer), None
    except ModuleNotFoundError as exc:
        category = "missing_target_dependency" if exc.name in ("peft", "transformers") else "missing_nested_dependency"
        return None, {"skipped": True, "category": category, "reason": f"ModuleNotFoundError: {exc.name}"}
    except Exception as exc:
        return None, {"skipped": True, "category": "incompatible_import", "reason": f"{type(exc).__name__}: {exc}"}


def check_against_peft(r, niter=None, device="cpu"):
    """在受控 SVD 随机流下核对 PEFT 的精确/Fast 初始化及前向，返回对齐报告。"""
    imports, skipped = import_peft()
    if skipped:
        return skipped
    LoraConfig, get_peft_model, LlamaConfig, LlamaForCausalLM, lora_layer = imports
    size = max(32, ((r+15)//8)*8)  # >= r+8; four heads with even head_dim
    cases = [(regime, width, rank, kind)
        for regime, width, rank in (("requested_rank", size, r), ("low_rank_fraction", 256, 16))
        for kind in ("gapped", "gaussian")]
    samples = []
    with seeded(0, device):
        for regime, size, rank, kind in cases:
            model = LlamaForCausalLM(LlamaConfig(hidden_size=size, intermediate_size=size*2,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4, vocab_size=32)).to(device=device, dtype=torch.float32)
            name = "model.layers.0.self_attn.q_proj"
            if kind == "gapped":
                with torch.no_grad():
                    model.get_submodule(name).weight.copy_(known_spectrum(size, size, rank)[0])
            w = model.get_submodule(name).weight.detach().clone()
            with seeded(17, device):
                ours = pissa_init(w, rank, niter=niter, **({"q": rank} if niter is not None else {}))
            config = LoraConfig(r=rank, lora_alpha=rank, target_modules=["q_proj"], lora_dropout=0.,
                init_lora_weights="pissa" if niter is None else f"pissa_niter_{niter}")
            original = lora_layer.svd_lowrank
            def controlled(*args, **kwargs):
                """为 lowrank SVD 临时隔离随机流，保证两套实现使用同一分解随机性。"""
                with seeded(17, device):
                    return original(*args, **kwargs)
            with patch.object(lora_layer, "svd_lowrank", controlled):
                adapted = get_peft_model(model, config)
            target = adapted.get_submodule("base_model.model."+name)
            pa, pb = target.lora_A["default"].weight, target.lora_B["default"].weight
            pd, od = pb @ pa, ours.lora_B @ ours.lora_A
            pr = target.base_layer.weight
            x = torch.randn(8, size, device=device)
            with torch.no_grad():
                py = target(x)
                oy = F.linear(x, ours.residual)+F.linear(F.linear(x, ours.lora_A), ours.lora_B)
            samples.append({"spectrum": kind, "shape": [size, size], "r": rank,
                "regime": regime, "r_over_k": rank/size,
                "q": rank if niter is not None else None,
                "passed": all([close(od, pd), close(ours.residual, pr), close(ours.residual+od, pr+pd), close(oy, py)]),
                "BA_rel_err": relative(od, pd), "residual_rel_err": relative(ours.residual, pr),
                "merged_rel_err": relative(ours.residual+od, pr+pd), "forward_rel_err": relative(oy, py)})
    return {"passed": all(x["passed"] for x in samples), "samples": samples, "niter": niter,
        "rng_control": "isolated at svd_lowrank entry; q unchanged", "rtol": ROUNDING_RTOL}


def check_contracts(device="cpu"):
    """检查 PiSSA 的精度、设备、初始映射及 alpha 不等于 r 时拒绝的契约。"""
    rows = []
    for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        with seeded(9, device):
            base = nn.Linear(12, 10, device=device, dtype=dtype)
            layer = PiSSALinear.from_linear(base, 3)
            x = torch.randn(4, 12, device=device, dtype=dtype)
        with torch.no_grad():
            reference, actual = base(x), layer(x)
        rtol = {torch.float16: .01, torch.bfloat16: .08}.get(dtype, ROUNDING_RTOL)
        rows.append({"dtype": str(dtype), "device": str(device), "forward_rel_err": relative(actual, reference),
            "passed": layer.lora_A.dtype == dtype and layer.residual.device == base.weight.device and
                       close(actual.float(), reference.float(), rtol=rtol)})
    try:
        PiSSALinear(8, 8, 2, alpha=4)
    except ValueError:
        alpha_rejected = True
    else:
        alpha_rejected = False
    return {"passed": all(x["passed"] for x in rows) and alpha_rejected, "samples": rows,
        "alpha_mismatch_rejected": alpha_rejected, "note": "storage dtype preserved; decomposition computes FP32"}


def main(argv=None):
    """解析验证参数，执行数学性质、Fast 质量、PEFT 对齐与契约检查，返回退出码。"""
    parser = ArgumentParser64(description="M0 mathematical, Fast quality and PEFT checks")
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--niter", type=int, default=None)
    parser.add_argument("--niters", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-peft", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if min(args.in_features, args.out_features, args.r, args.threads)<=0 or args.r>min(args.in_features, args.out_features):
        parser.error("positive dimensions/threads and 0 < r <= min(shape) required")
    if any(n<0 for n in args.niters) or (args.niter is not None and args.niter<0):
        parser.error("niter must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.niters)) != len(args.niters):
        parser.error("seeds and niters must be distinct")
    torch.set_num_threads(args.threads)
    config = {**vars(args), "output": str(args.output.resolve()), "storage_dtype": "torch.float32"}
    checks = {
        "reconstruction_and_forward": lambda: check_reconstruction(args.in_features, args.out_features, args.r, args.niter, args.device),
        "svd_property_eckart_young": lambda: check_svd_property(args.in_features, args.out_features, args.r, args.device),
        "fast_quality": lambda: check_fast_quality(args.in_features, args.out_features, args.r, args.niters, args.seeds, args.device),
        "fast_sensitivity": lambda: check_fast_sensitivity(args.in_features, args.out_features, args.r, args.seeds, args.device),
        "layer_contracts": lambda: check_contracts(args.device)}
    skipped = lambda: {"skipped": True, "reason": "--skip-peft", "category": "user_requested"}
    checks["peft_exact"] = skipped if args.skip_peft else lambda: check_against_peft(args.r, device=args.device)
    for niter in args.niters:
        checks[f"fast_reconstruction_and_forward_niter_{niter}"] = lambda n=niter: check_reconstruction(
            args.in_features, args.out_features, args.r, n, args.device)
        checks[f"peft_fast_niter_{niter}"] = skipped if args.skip_peft else lambda n=niter: check_against_peft(args.r, n, args.device)
    def lora_check():
        """按包导入或直接脚本模式加载 M0 的 LoRA/PEFT 对齐检查并执行。"""
        if __package__:
            from .mnist_toy import check_lora_peft
        else:
            from mnist_toy import check_lora_peft
        return check_lora_peft(args.device)
    checks["lora_peft"] = skipped if args.skip_peft else lora_check
    return run_checks(args.output / "metrics/verify.json", config, checks, args.device)


if __name__ == "__main__":
    sys.exit(main())
