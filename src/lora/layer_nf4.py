"""QLoRA 层：冻结的 ``Linear4bit`` 基座 + 可训练的 FP32（默认）LoRA 增量。

基座就是模型加载时已量化好的同一个 ``Linear4bit`` 子模块（不 clone、不复制打包权重）。
A 为 Kaiming 均匀初始化，B 为零，因此初始输出等于“实际量化后的基座”输出，
而不是量化前的 W。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adapters.base import LowRankAdapterBase, validate_rank_alpha
from adapters.quant import linear4bit_ready


class QLoRALinear(LowRankAdapterBase):
    """保留冻结的 4-bit 基座子模块，叠加可训练的低秩增量。"""

    def __init__(
        self,
        linear_4bit: nn.Module,
        r: int,
        alpha: float | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """校验基座并创建 Kaiming A 与零 B。

        Args:
            linear_4bit: 已量化就绪的 ``bnb.nn.Linear4bit``。
            r: 适配器秩（正整数，不受 min(in, out) 限制）。
            alpha: 缩放分子，None 取 r；alpha != r 对 LoRA 合法。
            dtype: A/B 存储精度。低层默认保持旧值 bfloat16；新协议由调用方显式传 float32。
            device: A/B 设备，None 时跟随基座。
            generator: 可选 CPU 随机源，仅用于 A 的初始化；None 时使用全局 RNG（既有行为）。

        Raises:
            TypeError: 基座不是量化就绪的 Linear4bit。
            ValueError: r/alpha/dtype 不合法。
        """
        if not linear4bit_ready(linear_4bit):
            raise TypeError(
                f"QLoRA 基座必须是已量化就绪的 Linear4bit，收到 {type(linear_4bit).__name__}"
            )
        validate_rank_alpha(r, alpha)
        super().__init__(
            linear_4bit, linear_4bit.in_features, linear_4bit.out_features, r, alpha, dtype=dtype, device=device,
        )
        # A 先在 CPU 上初始化再迁移，保持既有的 CPU RNG 来源。
        a_cpu = torch.empty(r, self.in_features, dtype=dtype)
        nn.init.kaiming_uniform_(a_cpu, a=5**0.5, generator=generator)
        with torch.no_grad():
            self.lora_A.copy_(a_cpu)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """拒绝直接合并 4-bit 基座；调用时始终抛出 NotImplementedError。"""
        raise NotImplementedError("QLoRA 不能合并 4-bit base weight")

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """拒绝折叠为普通线性层；调用时始终抛出 NotImplementedError。"""
        raise NotImplementedError("QLoRA 不能折叠为 nn.Linear")
