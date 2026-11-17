"""量化适配器共用的低秩适配器基类。

``LowRankAdapterBase`` 持有一个冻结的基座子模块（量化场景下是 bitsandbytes 的
``Linear4bit``）以及根层级的 ``lora_A`` / ``lora_B``，并实现两种量化适配器共同的前向。
它与 ``Linear4bit`` 是组合关系：基座的量化存储由 ``Linear4bit``/``Params4bit`` 管理，
本类不创建完整浮点残差，也不决定初始化方法或导出协议。

本模块导入时不加载 bitsandbytes；只有子类/注入入口按需导入。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def validate_rank_alpha(r, alpha) -> tuple[int, float]:
    """校验 r 与 alpha，返回 (r, alpha 浮点值)。

    r 必须是正整数（排除 bool）；alpha 为 None 时取 r，否则必须是有限正数。
    拒绝 alpha=0 是本项目的接口政策，不是数学上的限制。

    Raises:
        ValueError: r 或 alpha 不合法。
    """
    if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
        raise ValueError(f"r 必须是正整数，收到 r={r!r}")
    alpha_val = float(r) if alpha is None else float(alpha)
    if not math.isfinite(alpha_val) or alpha_val <= 0:
        raise ValueError(f"alpha 必须是有限的正数，收到 alpha={alpha!r}")
    return r, alpha_val


class LowRankAdapterBase(nn.Module):
    """冻结基座 + 可训练低秩增量：``y = base(x) + (alpha/r) * (x A^T) B^T``。

    Attributes:
        base: 冻结的基座子模块（不 clone、不 detach 其输出，输入梯度可经 base 回传）。
        lora_A: ``(r, in_features)`` 的 Parameter，精度为 ``dtype``。
        lora_B: ``(out_features, r)`` 的 Parameter，精度为 ``dtype``。
        scaling: ``alpha / r``。
    """

    def __init__(
        self,
        base: nn.Module,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        """冻结 base，创建全零的 A/B（具体初始化由子类写入）。

        Args:
            base: 已构造好的基座子模块；本构造器会把它的所有参数设为 requires_grad=False。
            in_features: 逻辑输入维度（不是 packed 存储形状）。
            out_features: 逻辑输出维度。
            r: 适配器秩。
            alpha: 缩放分子，None 取 r。
            dtype: A/B 的存储精度。
            device: A/B 所在设备，None 时跟随 base 的第一个参数。

        Raises:
            ValueError: r/alpha/dtype 不合法。
        """
        super().__init__()
        r, alpha_val = validate_rank_alpha(r, alpha)
        if dtype not in FLOAT_DTYPES:
            raise ValueError(f"适配器 dtype 必须是 {FLOAT_DTYPES} 之一，收到 {dtype}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r = r
        self.alpha = alpha_val
        self.scaling = alpha_val / r
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        if device is None:
            device = next(base.parameters()).device
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ``base(x) + scaling * B(A(x))``，输出 dtype 与 ``base(x)`` 一致。

        输入先转到 A 的存储精度再进入低秩分支（autocast 开启时 F.linear 仍按 autocast
        规则计算，存储 FP32 不等于每个算子都用 FP32）。base 前向不在 no_grad 内，
        冻结权重不切断输入梯度。
        """
        out = self.base(x)
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + (self.scaling * delta).to(out.dtype)

    def extra_repr(self) -> str:
        """返回 repr 中的维度、秩、缩放与适配器精度。"""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, "
            f"alpha={self.alpha}, scaling={self.scaling}, adapter_dtype={self.lora_A.dtype}"
        )
