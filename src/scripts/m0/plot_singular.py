"""M0：展示未量化权重与残差的奇异值谱及元素分布。

严格匹配目标键，真实输入失败不会静默变成随机输入。本地 safetensors 只读取
单个张量，不加载整个模型；不写论文文件。
"""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time

import torch

if __package__:
    from .verify_init import (ArgumentParser64, DEFAULT_OUTPUT, atomic_json, finish_report,
        new_report, seeded)
else:
    from verify_init import (ArgumentParser64, DEFAULT_OUTPUT, atomic_json, finish_report,
        new_report, seeded)
from pissa import pissa_init

MAX_HEADER = 16 * 1024 * 1024
_ST_DTYPE = {"F64": torch.float64, "F32": torch.float32,
             "F16": torch.float16, "BF16": torch.bfloat16}


def _base_url(source, repo):
    """按选定平台和仓库 ID 生成 safetensors 文件解析地址前缀。"""
    if source == "modelscope":
        return f"https://modelscope.cn/models/{repo}/resolve/master"
    if source == "hf":
        return f"https://huggingface.co/{repo}/resolve/main"
    raise ValueError(source)


def _filename(name):
    """校验远端分片名为安全相对路径，拒绝绝对路径及目录穿越。"""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe shard filename")
    return name


def _range_get(url, start, end, timeout=30, *, retries=2, etag_state=None, trace=None):
    """流式读取有界闭区间 [start, end]，只重试传输错误和临时 HTTP 错误。

    要求响应满足 Range 协议并受长度限制；记录请求轨迹与 ETag 一致性信息。
    """
    import requests
    if start < 0 or end < start or retries < 0:
        raise ValueError("invalid range/retry count")
    size = end-start+1
    for attempt in range(retries+1):
        response = None
        entry = {"url": url, "range": [start, end], "attempt": attempt+1}
        try:
            response = requests.get(url, headers={"Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity"}, timeout=timeout, stream=True)
            entry.update(status_code=response.status_code, etag=response.headers.get("ETag"),
                content_range=response.headers.get("Content-Range"),
                last_modified=response.headers.get("Last-Modified"))
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.ConnectionError(f"transient HTTP {response.status_code}")
            if response.status_code != 206:
                raise ValueError(f"Range requires HTTP 206, received {response.status_code}; body not read")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
            if not match or (int(match[1]), int(match[2])) != (start, end):
                raise ValueError("invalid Content-Range")
            if match[3] != "*" and int(match[3]) <= end:
                raise ValueError("range outside declared file length")
            encoding = response.headers.get("Content-Encoding", "identity")
            if encoding != "identity":
                raise ValueError("compressed range response is unsupported")
            if "Content-Length" in response.headers and int(response.headers["Content-Length"]) != size:
                raise ValueError("incorrect Content-Length")
            etag = response.headers.get("ETag")
            if etag_state is not None:
                if url in etag_state and etag_state[url] != etag:
                    raise ValueError("ETag changed between Range requests")
                if etag is not None:
                    etag_state[url] = etag
            chunks, length = [], 0
            for chunk in response.iter_content(chunk_size=min(size+1, 1024*1024)):
                length += len(chunk)
                if length > size:
                    raise ValueError("response exceeded requested length")
                chunks.append(chunk)
            if length != size:
                raise ValueError("truncated range response")
            return b"".join(chunks)
        except requests.RequestException as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                raise
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if response is not None:
                response.close()
            if trace is not None:
                trace.append(entry)


def _index_json(url, retries=2, trace=None):
    """读取并校验受限大小的 safetensors 索引；索引不存在时返回 None。"""
    import requests
    for attempt in range(retries+1):
        response = None
        entry = {"url": url, "attempt": attempt+1}
        try:
            response = requests.get(url, timeout=30, stream=True, headers={"Accept-Encoding": "identity"})
            entry.update(status_code=response.status_code, etag=response.headers.get("ETag"),
                         last_modified=response.headers.get("Last-Modified"))
            if response.status_code == 404:
                return None
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.ConnectionError(f"transient HTTP {response.status_code}")
            if response.status_code != 200:
                raise ValueError(f"index HTTP {response.status_code}")
            chunks, count = [], 0
            for chunk in response.iter_content(chunk_size=65536):
                count += len(chunk)
                if count > MAX_HEADER:
                    raise ValueError("index exceeds size limit")
                chunks.append(chunk)
            return json.loads(b"".join(chunks))
        except requests.RequestException as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                raise
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if response is not None:
                response.close()
            if trace is not None:
                trace.append(entry)


def _resolve_key(base, layer, *, retries=2, trace=None):
    """严格按指定权重键定位分片，不将未命中键替换为其他层。"""
    key = layer if layer.endswith(".weight") else layer+".weight"
    index = _index_json(base+"/model.safetensors.index.json", retries, trace)
    if index is None:
        return key, "model.safetensors"
    mapping = index["weight_map"]
    if key not in mapping:
        raise KeyError(f"requested tensor not in index: {key}")
    return key, _filename(mapping[key])


def _load_tensor_via_range(base, filename, key, *, retries=2, trace=None):
    """按 safetensors 头部定位单张量字节，读取并返回浮点权重及来源信息。"""
    url = f"{base}/{_filename(filename)}"
    state = {}
    get = lambda start, end: _range_get(url, start, end, retries=retries, etag_state=state, trace=trace)
    header_len = struct.unpack("<Q", get(0, 7))[0]
    if not 2 <= header_len <= MAX_HEADER:
        raise ValueError("safetensors header exceeds limit or is empty")
    header = json.loads(get(8, 8+header_len-1))
    if key not in header:
        raise KeyError(f"requested tensor not in safetensors header: {key}")
    meta = header[key]
    dtype, shape = _ST_DTYPE.get(meta["dtype"]), meta["shape"]
    if dtype is None or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("expected nonempty 2D floating weight")
    offsets = meta["data_offsets"]
    if len(offsets) != 2 or any(type(n) is not int for n in offsets):
        raise ValueError("invalid tensor offsets")
    begin, end = offsets
    expected = math.prod(shape)*torch.empty((), dtype=dtype).element_size()
    if begin < 0 or end-begin != expected:
        raise ValueError("tensor shape/dtype/offset length mismatch")
    raw = get(8+header_len+begin, 8+header_len+end-1)
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    return tensor.float(), {"original_dtype": str(dtype), "actual_key": key,
        "shard": filename, "url": url, "etag": state.get(url), "data_offsets": offsets}


def load_local_weight(path, key):
    """从本地 safetensors 单文件或分片目录读取指定张量，返回权重及来源信息。"""
    from safetensors import safe_open
    path = Path(path).resolve()
    key = key if key.endswith(".weight") else key+".weight"
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if key not in mapping:
                raise KeyError(f"requested tensor not in local index: {key}")
            filename = _filename(mapping[key])
            shard = (path / filename).resolve()
            if not shard.is_relative_to(path):
                raise ValueError("shard escapes local model directory")
        else:
            shard = path / "model.safetensors"
    else:
        shard = path
    if not shard.is_file():
        raise FileNotFoundError(f"local safetensors not found: {shard}")
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"requested tensor not in local shard: {key}")
        weight = f.get_tensor(key)
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("expected a 2D floating weight")
    return weight.float(), {"mode": "local", "actual_model": str(path), "actual_key": key,
        "requested_key": key, "shard": str(shard), "source": "local_safetensors",
        "original_dtype": str(weight.dtype), "revision": None, "etag": None}


def load_weight(args):
    """按随机、本地或远端模式加载单层权重，返回张量与可追溯来源。

    远端失败仅在显式允许时使用随机回退，并记录全部失败；不把随机输入当模型证据。
    """
    if args.random:
        with seeded(args.seed):
            weight = torch.randn(args.out_features, args.in_features)
        return weight, {"mode": "random", "actual_model": None, "actual_key": None,
            "source": "seeded_gaussian", "seed": args.seed, "original_dtype": "torch.float32"}
    if args.local_path is not None:
        return load_local_weight(args.local_path, args.layer)
    failures, trace = [], []
    sources = ["modelscope", "hf"] if args.source == "auto" else [args.source]
    for source in sources:
        try:
            base = _base_url(source, args.model)
            key, filename = _resolve_key(base, args.layer, retries=args.retries, trace=trace)
            weight, meta = _load_tensor_via_range(base, filename, key, retries=args.retries, trace=trace)
            meta.update(mode="remote", actual_model=args.model, requested_key=args.layer,
                source=source, requests=trace, prior_source_failures=failures, revision=None)
            return weight, meta
        except Exception as exc:
            failures.append({"source": source, "type": type(exc).__name__, "message": str(exc)})
    if not args.allow_random_fallback:
        raise RuntimeError(json.dumps({"failures": failures, "requests": trace}))
    with seeded(args.seed):
        weight = torch.randn(args.out_features, args.in_features)
    return weight, {"mode": "random_fallback", "actual_model": None, "actual_key": None,
        "source": "seeded_gaussian", "seed": args.seed, "original_dtype": "torch.float32",
        "failures": failures, "requests": trace}


def main(argv=None):
    """绘制未量化权重和残差的谱及元素分布，保存来源报告与图片，返回退出码。

    Args:
        argv: 可选参数列表；None 时读取进程参数。真实权重模式可能发起网络请求。

    Returns:
        成功为 0，执行异常为 1；非法参数由解析器以 64 退出。
    """
    parser = ArgumentParser64(description="Strict W/residual spectra and element distributions")
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--local-path", type=Path)
    parser.add_argument("--source", choices=["auto", "modelscope", "hf"], default="auto")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--allow-random-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r", type=int, default=128)
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if min(args.r, args.in_features, args.out_features, args.threads)<=0 or args.retries<0:
        parser.error("dimensions/r/threads must be positive; retries nonnegative")
    if args.random and args.local_path is not None:
        parser.error("--random and --local-path are mutually exclusive")
    if args.random and args.r>min(args.in_features, args.out_features):
        parser.error("r exceeds random matrix dimensions")
    if args.layer is None:
        args.layer = "model.layers.0.self_attn.q_proj" if args.local_path is not None else "model.language_model.layers.11.self_attn.q_proj"
    torch.set_num_threads(args.threads)
    config = {**vars(args), "output": str(args.output.resolve()),
        "local_path": str(args.local_path.resolve()) if args.local_path is not None else None}
    started = time.perf_counter()
    path = args.output / "metrics/singular.json"
    report = new_report("M0/plot_singular", config, args.device)
    atomic_json(path, report)
    try:
        tick = time.perf_counter()
        weight, provenance = load_weight(args)
        report["provenance"] = provenance
        report["timings_seconds"]["load_weight"] = time.perf_counter()-tick
        report["config"]["weight_shape"] = list(weight.shape)
        atomic_json(path, report)
        if args.r>min(weight.shape) or not bool(torch.isfinite(weight).all()):
            raise ValueError("invalid rank or non-finite weight")
        tick = time.perf_counter()
        init = pissa_init(weight.to(args.device), args.r)
        s_full = init.singular_values.double().cpu().numpy()
        s_res = torch.linalg.svdvals(init.residual.double()).cpu().numpy()
        report["timings_seconds"]["decomposition_and_residual_spectrum"] = time.perf_counter()-tick
        import numpy as np
        w, res = weight.cpu().numpy(), init.residual.float().cpu().numpy()
        low, high = min(float(w.min()), float(res.min())), max(float(w.max()), float(res.max()))
        if low == high:
            low, high = low-1., high+1.
        bins = np.linspace(low, high, 65)
        wc, _ = np.histogram(w, bins)
        rc, _ = np.histogram(res, bins)
        def stats(array):
            """计算数组的均值、总体标准差、最大绝对值及元素数，转换为 JSON 兼容类型。"""
            return {"mean": float(np.mean(array, dtype=np.float64)),
                "std_population": float(np.std(array, dtype=np.float64)),
                "max_abs": float(np.max(np.abs(array))), "elements": int(array.size)}
        energy = float(np.square(s_full).sum())
        report["results"] = {"W_singular_values": s_full.tolist(), "W_res_singular_values": s_res.tolist(),
            "W_singular_max": float(s_full.max()), "W_res_singular_max": float(s_res.max()),
            "principal_energy_ratio": float(np.square(s_full[:args.r]).sum())/energy if energy else None,
            "histogram_bins": bins.tolist(), "W_histogram_counts": wc.tolist(), "W_res_histogram_counts": rc.tolist(),
            "W_statistics": stats(w), "W_res_statistics": stats(res)}
        atomic_json(path, report)
        tick = time.perf_counter()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(range(1, len(s_full)+1), s_full)
        axes[0, 0].axvline(args.r, color="r", ls="--", label=f"r={args.r}")
        axes[0, 0].legend()
        axes[0, 1].plot(range(1, len(s_res)+1), s_res, color="tab:orange")
        for ax, title in zip(axes[0], ("W singular values", "Residual singular values")):
            ax.set(title=title, xlabel="Singular-value index (1-based)", ylabel="Magnitude")
        for ax, counts, title in zip(axes[1], (wc, rc), ("W elements", "Residual elements")):
            ax.stairs(counts, bins)
            ax.set(title=title, xlabel="Weight value (shared bins)", ylabel="Element count", xlim=(low, high))
        actual = provenance.get("actual_model")
        title = f"{provenance['mode']} | {Path(actual).name if actual else 'RANDOM MATRIX — not model evidence'}"
        fig.suptitle(title+f"\n{provenance.get('actual_key') or 'no model layer'} | r={args.r}")
        fig.tight_layout()
        figure = args.output / "figures/singular_values.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=150)
        plt.close(fig)
        report["config"]["figure"] = str(figure.resolve())
        report["timings_seconds"]["plot"] = time.perf_counter()-tick
        code = finish_report(report, path, "passed", 0, started)
        print(f"任务已成功完成，完整 singular.json 已保存：{path.resolve()}", flush=True)
        return code
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finish_report(report, path, "error", 1, started)
        print(f"Spectrum error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
