"""把模型中的目标 nn.Linear 替换成 PiSSALinear，以及训练后折叠回标准权重。"""

from __future__ import annotations

import torch.nn as nn

from .layer import PiSSALinear

# Llama / Qwen / Gemma 等主流 dense 解码器的全部线性层（注意力 + MLP）。
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _match(name: str, targets: tuple[str, ...]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf in targets


def inject_pissa(
    model: nn.Module,
    r: int,
    alpha: float | None = None,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    niter: int | None = None,
) -> list[str]:
    """就地把命中 ``target_modules`` 的 nn.Linear 替换为 PiSSALinear。

    返回被替换模块的名字列表，便于核对覆盖范围。
    """
    replaced: list[str] = []
    # 先收集，避免遍历过程中修改子模块结构。
    to_replace = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _match(name, target_modules)
    ]
    for name, module in to_replace:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child_name = name.rsplit(".", 1)[-1]
        pissa_layer = PiSSALinear.from_linear(module, r=r, alpha=alpha, niter=niter)
        pissa_layer = pissa_layer.to(module.weight.device, module.weight.dtype)
        setattr(parent, child_name, pissa_layer)
        replaced.append(name)
    return replaced


def merge_pissa(model: nn.Module) -> list[str]:
    """把模型中所有 PiSSALinear 折叠回标准 nn.Linear，返回被折叠的名字列表。"""
    merged: list[str] = []
    to_merge = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PiSSALinear)
    ]
    for name, module in to_merge:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child_name = name.rsplit(".", 1)[-1]
        linear = module.to_linear().to(module.lora_A.device, module.lora_A.dtype)
        setattr(parent, child_name, linear)
        merged.append(name)
    return merged


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
