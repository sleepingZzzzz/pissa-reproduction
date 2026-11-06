"""按目标层名称注入与合并普通 LoRA。"""
from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .layer import LoRALinear


def _match(name: str, targets: tuple[str, ...]) -> bool:
    """判断模块名称最后一段是否精确命中 targets。"""
    return name.rsplit(".", 1)[-1] in targets


def inject_lora(model: nn.Module, r: int, alpha, target_modules) -> list[str]:
    """原地将命中的 nn.Linear 替换为普通 LoRALinear。

    Args:
        model: 待修改的模型。
        r: 适配器秩。
        alpha: 缩放分子；None 时取 r。
        target_modules: 按模块叶名称匹配的目标序列。

    Returns:
        按遍历顺序记录的被替换模块名称列表。

    保留既有的 (名称, 模块) 预收集方式，不包含 PiSSA 注入器的预校验改进。
    """
    replaced: list[str] = []
    to_replace = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and _match(n, tuple(target_modules))
    ]
    for name, module in tqdm(to_replace, desc="LoRA 注入", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        layer = LoRALinear(module, r=r, alpha=alpha).to(module.weight.device, module.weight.dtype)
        setattr(parent, child, layer)
        replaced.append(name)
    return replaced


def merge_lora(model: nn.Module) -> list[str]:
    """原地将所有 LoRALinear 合并成设备、精度对应的 nn.Linear。

    Args:
        model: 含普通 LoRA 包装层的模型。

    Returns:
        被合并的模块名称列表；合并后 A/B 不再作为独立参数存在。
    """
    merged: list[str] = []
    to_merge = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, module in tqdm(to_merge, desc="合并 LoRA", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        lin = module.to_linear().to(module.lora_A.device, module.lora_A.dtype)
        setattr(parent, child, lin)
        merged.append(name)
    return merged
