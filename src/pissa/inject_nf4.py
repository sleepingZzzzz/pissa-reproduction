"""既有 QPiSSA 注入；按需加载量化依赖，本次仅迁移。"""
from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .inject import _match


def inject_qpissa(model: nn.Module, r: int, alpha, target_modules, niter=None) -> list[str]:
    """原地将目标普通线性层替换为既有的 NF4 残差 PiSSA 层。

    Args:
        model: 待修改的模型。
        r: 适配器秩。
        alpha: PiSSA 缩放分子，受层构造器的 alpha == r 限制。
        target_modules: 匹配模块叶名称的目标序列。
        niter: None 为精确 SVD，整数为随机低秩 SVD 的幂迭代次数。

    Returns:
        被替换的模块名称列表。

    调用时才导入量化层；保留原有预收集方式和每八层清理 CUDA 缓存的行为。
    量化核心改进留到第四阶段，本次只迁移该函数。
    """
    from pissa.layer_nf4 import NFPiSSALinear

    replaced: list[str] = []
    to_replace = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and _match(n, tuple(target_modules))
    ]
    for name, module in tqdm(to_replace, desc="QPiSSA NF4 注入", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        layer = NFPiSSALinear.from_linear(module, r=r, alpha=alpha, niter=niter).to(
            module.weight.device, module.weight.dtype
        )
        setattr(parent, child, layer)
        del module  # 释放旧 Linear 层
        replaced.append(name)
        # 每 8 层清理一次 CUDA 缓存，防止 SVD 临时 tensor 累积
        if len(replaced) % 8 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    return replaced
