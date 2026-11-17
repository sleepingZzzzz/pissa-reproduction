"""QPiSSA 的注入与合并；量化依赖按需导入，普通 PiSSA 路径不受影响。"""
from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .inject import _check_targets


def inject_qpissa(
    model: nn.Module,
    r: int,
    alpha,
    target_modules,
    niter=None,
    *,
    rounds: int = 1,
    compute_dtype: torch.dtype = torch.bfloat16,
    double_quant: bool = True,
    init_double_quant: bool = False,
    adapter_dtype: torch.dtype = torch.float32,
    base_dtype: torch.dtype = torch.bfloat16,
    device=None,
    seed: int | None = None,
    reports: dict | None = None,
) -> list[str]:
    """原地将目标浮点 Linear 替换为 NFPiSSALinear（Linear4bit 管理的量化残差基座）。

    先整体校验目标类型与共享模块，然后只保存层名、逐层取对象并替换，使旧层在替换后
    可被回收；不靠 empty_cache 掩盖仍然存活的引用。SVD/量化中途失败不保证回滚。

    Args:
        model: 待修改模型，目标层须是未量化浮点 ``nn.Linear``。
        r: 适配器秩。
        alpha: PiSSA 缩放分子，须等于 r。
        target_modules: 匹配模块叶名称的目标序列。
        niter: Fast SVD 幂迭代次数（None 为精确 SVD）。
        rounds: 外层交替轮数。
        compute_dtype / double_quant / init_double_quant / adapter_dtype / base_dtype / seed:
            见 ``NFPiSSALinear.from_linear``。
        device: 目标 CUDA 设备，None 时取各层源权重设备。
        reports: 非 None 时写入 ``{层名: init_report}``。

    Returns:
        被替换模块的完整名称列表。
    """
    from pissa.layer_nf4 import NFPiSSALinear

    names = _check_targets(model, tuple(target_modules))
    replaced: list[str] = []
    for name in tqdm(names, desc="QPiSSA 交替初始化", unit="layer"):
        module = model.get_submodule(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        layer = NFPiSSALinear.from_linear(
            module, r=r, alpha=alpha, rounds=rounds, niter=niter, compute_dtype=compute_dtype,
            double_quant=double_quant, init_double_quant=init_double_quant, adapter_dtype=adapter_dtype,
            base_dtype=base_dtype, device=device, seed=seed,
        )
        setattr(parent, child, layer)
        if reports is not None:
            reports[name] = layer.init_report
        del module, layer
        replaced.append(name)
    return replaced


def merge_qpissa(model: nn.Module, dtype: torch.dtype = torch.float32) -> list[str]:
    """原地把全部 NFPiSSALinear 折叠为普通 ``nn.Linear``（``dequant(base) + scaling*B@A``）。

    Args:
        model: 含 QPiSSA 层的模型。
        dtype: 合并后线性层的精度，默认 FP32，后续由调用方统一转换保存精度。

    Returns:
        被合并的模块名称列表。
    """
    from pissa.layer_nf4 import NFPiSSALinear

    names = [n for n, m in model.named_modules() if isinstance(m, NFPiSSALinear)]
    merged: list[str] = []
    for name in tqdm(names, desc="合并 QPiSSA", unit="layer"):
        module = model.get_submodule(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        setattr(parent, child, module.to_linear(dtype))
        del module
        merged.append(name)
    return merged
