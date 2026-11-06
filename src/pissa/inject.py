"""把模型中的目标 nn.Linear 替换成 PiSSALinear，以及训练后折叠回标准权重。"""

from __future__ import annotations

import torch.nn as nn

from .layer import PiSSALinear

try:  # 进度条为可选依赖，缺失时退化为直通迭代器。
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):
        """缺少 tqdm 时原样返回迭代对象，忽略进度条参数。"""
        return it

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
    """按模块名称最后一段精确匹配目标名称。"""
    leaf = name.rsplit(".", 1)[-1]
    return leaf in targets


def _check_targets(model: nn.Module, targets: tuple[str, ...]) -> list[str]:
    """在修改模型前收集并校验全部目标线性层名称。

    Args:
        model: 待检查的模型，不在此函数中修改。
        targets: 模块叶名称元组。

    Returns:
        待替换模块的完整名称列表。

    Raises:
        TypeError: 目标是已知量化层或权重不是浮点张量。
        ValueError: 同一目标 Linear 对象注册为多个名称。

    共享模块检查针对对象别名，不代表检测任意不同模块之间的共享权重。
    """
    names_by_id: dict[int, list[str]] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, nn.Linear):
            names_by_id.setdefault(id(module), []).append(name)

    names: list[str] = []
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and _match(name, targets)):
            continue
        cls_names = {c.__name__ for c in type(module).__mro__}
        if cls_names & {"Linear4bit", "Linear8bitLt"} or not module.weight.is_floating_point():
            raise TypeError(
                f"{name}: 不能对量化层或非浮点权重做 PiSSA 注入 "
                f"(class={type(module).__name__}, weight.dtype={module.weight.dtype})"
            )
        aliases = names_by_id[id(module)]
        if len(aliases) > 1:
            raise ValueError(f"{name}: 该层被多个名字共享 {aliases}，PiSSA 注入不支持权重共享")
        names.append(name)
    return names


def inject_pissa(
    model: nn.Module,
    r: int,
    alpha: float | None = None,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    niter: int | None = None,
) -> list[str]:
    """原地将目标 nn.Linear 替换为 PiSSALinear，并逐层释放旧引用。

    Args:
        model: 待修改的模型。
        r: 适配器秩。
        alpha: None 或 r，受 PiSSALinear 的范围限制。
        target_modules: 要匹配的模块叶名称，默认包含注意力及 MLP 的七种投影。
        niter: None 为精确 SVD，整数为 Fast SVD 幂迭代次数。

    Returns:
        被替换模块的完整名称列表。源层的设备和精度由 from_linear 保留。

    目标类型与共享模块先整体校验；后续初始化失败时不保证回滚此前替换。
    """
    replaced: list[str] = []
    # 只保存名字，不保存模块对象：每层替换后旧 Linear 立即可被回收，
    # 避免峰值显存变成"整个旧模型 + 整个新模型"。
    names = _check_targets(model, target_modules)
    for name in tqdm(names, desc="PiSSA SVD 初始化", unit="layer"):
        module = model.get_submodule(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child_name = name.rsplit(".", 1)[-1]
        pissa_layer = PiSSALinear.from_linear(module, r=r, alpha=alpha, niter=niter)
        pissa_layer = pissa_layer.to(module.weight.device, module.weight.dtype)
        setattr(parent, child_name, pissa_layer)
        del module
        replaced.append(name)
    return replaced


def merge_pissa(model: nn.Module) -> list[str]:
    """原地将全部 PiSSALinear 折叠为普通线性层，返回被合并的名称列表。

    Args:
        model: 含 PiSSA 层的模型；合并后保留各层设备和精度。

    Returns:
        按遍历顺序记录的被合并模块名称，不再保留独立 A/B。
    """
    merged: list[str] = []
    # 同 inject_pissa：只存名字，逐层释放旧模块。
    names = [name for name, module in model.named_modules() if isinstance(module, PiSSALinear)]
    for name in tqdm(names, desc="合并 PiSSA", unit="layer"):
        module = model.get_submodule(name)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child_name = name.rsplit(".", 1)[-1]
        linear = module.to_linear().to(module.lora_A.device, module.lora_A.dtype)
        setattr(parent, child_name, linear)
        del module
        merged.append(name)
    return merged


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (可训练 Parameter 元素数, 全部 Parameter 元素数)。

    Args:
        model: 待统计的模型。

    Returns:
        两个整数；均不计 buffer，注入后的第二项不能当作原始基座参数量。
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
