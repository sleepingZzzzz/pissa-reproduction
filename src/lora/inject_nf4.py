"""QLoRA 注入与解包；bitsandbytes 在函数内按需导入。"""
from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from .layer_nf4 import QLoRALinear
from .inject import _match


def _check_qlora_targets(model: nn.Module, targets: tuple[str, ...]) -> list[str]:
    """在修改模型前收集并校验全部 QLoRA 目标层名称。

    Raises:
        TypeError: 命中层不是已量化就绪的 Linear4bit（模型未按 4-bit 加载，或已被包装）。
        ValueError: 同一层对象以多个名字注册。
    """
    from adapters.quant import linear4bit_ready

    names_by_id: dict[int, list[str]] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, nn.Linear):
            names_by_id.setdefault(id(module), []).append(name)
    names: list[str] = []
    bad: list[str] = []
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and _match(name, targets)):
            continue
        if not linear4bit_ready(module):
            bad.append(name)
            continue
        aliases = names_by_id[id(module)]
        if len(aliases) > 1:
            raise ValueError(f"{name}: 该层被多个名字共享 {aliases}，QLoRA 注入不支持权重共享")
        names.append(name)
    if bad:
        raise TypeError(
            f"QLoRA 目标层不是已量化就绪的 Linear4bit（模型未按 4-bit 加载？）：{bad[:3]} …共 {len(bad)} 层"
        )
    return names


def inject_lora_4bit(
    model: nn.Module,
    r: int,
    alpha,
    target_modules,
    dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> list[str]:
    """原地为目标 Linear4bit 添加 QLoRA，不复制打包的基座权重。

    先整体校验目标，再只保存层名、逐层替换。

    Args:
        model: 已按 4-bit 加载的模型。
        r: 适配器秩。
        alpha: 缩放分子。
        target_modules: 匹配模块叶名称的目标序列。
        dtype: A/B 存储精度。低层默认保持旧值 bfloat16；新协议由调用方显式传 float32。
        generator: 可选 CPU 随机源，用于 A 的 Kaiming 初始化。

    Returns:
        被包装的模块名称列表；没有命中目标时返回空列表（由调用方的覆盖检查报错）。
    """
    names = _check_qlora_targets(model, tuple(target_modules))
    replaced: list[str] = []
    for name in tqdm(names, desc="QLoRA 注入", unit="layer"):
        module = model.get_submodule(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        setattr(parent, child, QLoRALinear(module, r=r, alpha=alpha, dtype=dtype, generator=generator))
        del module
        replaced.append(name)
    return replaced


def unwrap_qlora(model: nn.Module, adapter_sd: dict[str, torch.Tensor] | None = None) -> list[str]:
    """原地移除 QLoRA 包装，恢复原始基座子模块，可选导出适配器。

    Args:
        model: 含 QLoRALinear 的模型。
        adapter_sd: 非 None 时写入以完整模块名为前缀的 A/B CPU 副本（直接从设备拷到 CPU，
            不先在 GPU 上 clone）。

    Returns:
        被解包的模块名称列表；基座没有吸收 A/B 增量。
    """
    names = [n for n, m in model.named_modules() if isinstance(m, QLoRALinear)]
    unwrapped: list[str] = []
    for name in tqdm(names, desc="解包 QLoRA", unit="layer"):
        module = model.get_submodule(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        if adapter_sd is not None:
            adapter_sd[f"{name}.lora_A"] = module.lora_A.detach().to("cpu", copy=True)
            adapter_sd[f"{name}.lora_B"] = module.lora_B.detach().to("cpu", copy=True)
        setattr(parent, child, module.base)
        del module
        unwrapped.append(name)
    return unwrapped
