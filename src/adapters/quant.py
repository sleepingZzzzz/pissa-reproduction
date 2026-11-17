"""NF4 量化的小型辅助函数：往返量化、Linear4bit 构造与反量化。

bitsandbytes 只在函数内部按需导入，普通 LoRA/PiSSA 路径不会加载它。
所有函数固定 NF4、blocksize=64（bitsandbytes 默认），不引入其他量化类型。
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

NF4_BLOCKSIZE = 64


@contextlib.contextmanager
def local_rng(seed: int | None, device: torch.device | str):
    """在上下文内临时固定 CPU 与目标 CUDA 设备的 RNG，退出时恢复，不影响全局随机流。

    seed 为 None 时不做任何事（保持既有全局 RNG 行为）。不重设整个程序的全局 seed。
    """
    if seed is None:
        yield
        return
    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def nf4_roundtrip(weight: torch.Tensor, compress_statistics: bool = False) -> torch.Tensor:
    """NF4 量化再反量化，返回与输入同形状、同 dtype 的张量。

    Args:
        weight: 二维权重，不原地修改。
        compress_statistics: 是否对 absmax 做二次（嵌套）量化，即 double quant。

    量化在输入所在设备、输入 dtype 上进行（初始化阶段应传 FP32）。
    """
    import bitsandbytes as bnb

    w = weight.detach().contiguous()
    packed, state = bnb.functional.quantize_4bit(
        w, blocksize=NF4_BLOCKSIZE, compress_statistics=compress_statistics, quant_type="nf4"
    )
    return bnb.functional.dequantize_4bit(packed, state).to(weight.dtype).reshape(weight.shape)


def build_linear4bit(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    compute_dtype: torch.dtype,
    double_quant: bool,
    device: torch.device | str,
) -> nn.Module:
    """用浮点权重构造并量化一个 ``bnb.nn.Linear4bit``（NF4，blocksize 64）。

    量化发生在 ``weight`` 当前的 dtype 上（例如传 BF16 即在 BF16 上量化，对应官方训练加载阶段）。
    构造在 meta 设备上完成，不分配随机初始权重，也不推进全局 RNG。

    Args:
        weight: ``(out, in)`` 浮点权重，不原地修改。
        bias: 可选 ``(out,)`` 冻结偏置。
        compute_dtype: Linear4bit 的计算精度。
        double_quant: 是否对 absmax 做嵌套量化。
        device: 目标 CUDA 设备（bitsandbytes 在迁移到 CUDA 时执行量化）。

    Returns:
        已量化、参数 requires_grad=False 的 Linear4bit。

    Raises:
        RuntimeError: 目标不是 CUDA 设备，或量化未就绪。
    """
    from bitsandbytes.nn import Linear4bit, Params4bit

    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(f"Linear4bit 需要 CUDA 设备，收到 {dev}")
    out_f, in_f = weight.shape
    with torch.device("meta"):
        lin = Linear4bit(
            in_f, out_f, bias=bias is not None, compute_dtype=compute_dtype,
            compress_statistics=double_quant, quant_type="nf4",
        )
    lin.weight = Params4bit(
        data=weight.detach().to(dev).contiguous(), requires_grad=False, blocksize=NF4_BLOCKSIZE,
        compress_statistics=double_quant, quant_type="nf4", module=lin,
    )
    lin.bias = nn.Parameter(bias.detach().to(dev).clone(), requires_grad=False) if bias is not None else None
    lin = lin.to(dev)  # 触发 Params4bit 的 CUDA 量化
    if not linear4bit_ready(lin):
        raise RuntimeError("Linear4bit 量化未就绪（quant_state 缺失）。")
    return lin


def linear4bit_ready(module: nn.Module) -> bool:
    """判断 module 是否为已量化就绪的 Linear4bit（权重 uint8 且带 quant_state）。"""
    from bitsandbytes.nn import Linear4bit

    if not isinstance(module, Linear4bit):
        return False
    w = module.weight
    return getattr(w, "quant_state", None) is not None and w.dtype == torch.uint8


def dequantize_linear4bit(module: nn.Module) -> torch.Tensor:
    """反量化 Linear4bit 的权重，返回 ``(out, in)`` 张量，dtype 为 quant_state.dtype。"""
    import bitsandbytes as bnb

    if not linear4bit_ready(module):
        raise RuntimeError("基座不是已就绪的 Linear4bit。")
    return bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state).reshape(
        module.out_features, module.in_features
    )
