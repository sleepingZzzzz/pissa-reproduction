"""把模型中的目标 nn.Linear 替换成 PiSSALinear，以及训练后折叠回标准权重。"""

from __future__ import annotations

import torch.nn as nn

from .layer import PiSSALinear

try:  # 进度条为可选依赖，缺失时退化为直通迭代器。
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):
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
    leaf = name.rsplit(".", 1)[-1]
    return leaf in targets


def _check_targets(model: nn.Module, targets: tuple[str, ...]) -> list[str]:
    """收集并校验待替换的层名；校验失败时直接报错（此时模型尚未被改动）。

    - 拒绝量化层（bitsandbytes 的 Linear4bit / Linear8bitLt）与非浮点权重：
      对打包的 uint8 权重做 SVD 没有意义。
    - 拒绝共享层：named_modules() 默认按对象去重，同一个 Linear 以多个名字注册时，
      只有一个名字会被替换，其余别名仍指向旧层，模型会静默变成两套权重。
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
    """就地把命中 ``target_modules`` 的 nn.Linear 替换为 PiSSALinear。

    返回被替换模块的名字列表，便于核对覆盖范围。
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
    """把模型中所有 PiSSALinear 折叠回标准 nn.Linear，返回被折叠的名字列表。"""
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
    """返回 (可训练参数量, 总参数量)。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
