"""既有 QLoRA 注入与解包；本次仅迁移。"""
from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .layer_nf4 import QLoRALinear
from .inject import _match


def inject_lora_4bit(model: nn.Module, r: int, alpha, target_modules) -> list[str]:
    """原地为目标 Linear4bit 添加 QLoRA，不复制打包的基座权重。

    Args:
        model: 已按 4-bit 加载的模型。
        r: 适配器秩。
        alpha: 缩放分子。
        target_modules: 匹配模块叶名称的目标序列。

    Returns:
        被包装的模块名称列表。

    Raises:
        TypeError: 匹配到的线性层继承链中没有名为 Linear4bit 的类。
    """
    replaced: list[str] = []
    to_replace = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and _match(n, tuple(target_modules))
    ]
    # QLoRA 的基座必须已被 bitsandbytes 量化为 Linear4bit；匹配到普通 nn.Linear 说明
    # 模型没有按 4-bit 加载，此时静默包装会得到"未量化的 QLoRA"，直接报错。
    not_4bit = [n for n, m in to_replace if "Linear4bit" not in {c.__name__ for c in type(m).__mro__}]
    if not_4bit:
        raise TypeError(f"QLoRA 目标层不是 Linear4bit（模型未按 4-bit 加载？）：{not_4bit[:3]} …共 {len(not_4bit)} 层")
    for name, module in tqdm(to_replace, desc="QLoRA 注入", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        # 4-bit Linear4bit → QLoRALinear wrapper（不 clone uint8 weight）
        layer = QLoRALinear(module, r=r, alpha=alpha).to(module.weight.device)
        setattr(parent, child, layer)
        replaced.append(name)
    return replaced


def unwrap_qlora(model: nn.Module, adapter_sd: dict[str, torch.Tensor] | None = None) -> list[str]:
    """原地移除 QLoRA 包装，恢复原始基座子模块，可选导出适配器。

    Args:
        model: 含 QLoRALinear 的模型。
        adapter_sd: 非 None 时写入以完整模块名为前缀的 A/B CPU 克隆张量。

    Returns:
        被解包的模块名称列表；基座没有吸收 A/B 增量。
    """
    unwrapped: list[str] = []
    to_unwrap = [(n, m) for n, m in model.named_modules() if isinstance(m, QLoRALinear)]
    for name, module in tqdm(to_unwrap, desc="解包 QLoRA", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        if adapter_sd is not None:
            adapter_sd[f"{name}.lora_A"] = module.lora_A.data.clone().cpu()
            adapter_sd[f"{name}.lora_B"] = module.lora_B.data.clone().cpu()
        setattr(parent, child, module.base)
        unwrapped.append(name)
    return unwrapped
