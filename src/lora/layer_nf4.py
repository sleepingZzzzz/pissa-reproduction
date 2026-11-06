"""既有 QLoRA 包装层；本次仅迁移，不改变量化实现。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLoRALinear(nn.Module):
    """保留冻结的 4-bit 基座子模块，叠加可训练的普通精度 LoRA 增量。"""

    def __init__(self, linear_4bit: nn.Linear, r: int, alpha: float | None = None, dtype=torch.bfloat16):
        """保存并冻结原基座子模块，创建 Kaiming A 和零 B。

        Args:
            linear_4bit: 已量化基座层；此构造器自身不检查是否为 Linear4bit。
            r: 适配器秩。
            alpha: 缩放分子，None 时取 r。
            dtype: A/B 精度，默认为 bfloat16；注入调用方负责迁移设备。
        """
        super().__init__()
        self.in_features = linear_4bit.in_features
        self.out_features = linear_4bit.out_features
        self.r = r
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = self.alpha / r
        # 保持 4-bit base 层作为 frozen 子模块（不 clone weight）
        self.base = linear_4bit
        for p in self.base.parameters():
            p.requires_grad_(False)
        # LoRA A/B 使用计算精度
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 base(x) 与 (alpha/r) * xA.T B.T 相加，返回 (..., out) 张量。"""
        base = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """拒绝直接合并 4-bit 基座；调用时始终抛出 NotImplementedError。"""
        raise NotImplementedError("QLoRA 不能合并 4-bit base weight")

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """拒绝折叠为普通线性层；调用时始终抛出 NotImplementedError。"""
        raise NotImplementedError("QLoRA 不能折叠为 nn.Linear")
