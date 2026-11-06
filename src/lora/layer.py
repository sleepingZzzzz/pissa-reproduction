"""M1 LoRA 线性层：冻结基座，训练 Kaiming 初始化的 A 与零初始化的 B。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """冻结原线性映射，使用 Kaiming A、零 B 学习低秩增量。

    权重为 (out, in) 的 buffer；可训练 A 为 (r, in)，B 为 (out, r)。
    前向为 xW.T + bias + (alpha/r) * xA.T B.T，初始增量为零。
    本类保留 M1 的既有初始化，与 M0 的高斯初始化 LoRA 分开维护。
    """
    def __init__(self, linear: nn.Linear, r: int, alpha: float | None = None) -> None:
        """复制并冻结基座权重和偏置，创建可训练低秩参数。

        Args:
            linear: 源 nn.Linear；本函数不原地修改源层。
            r: 低秩维度；本类沿用既有实现，不额外校验取值。
            alpha: 缩放分子，None 时取 r，允许与 r 不同。

        A/B 使用源权重精度但先在默认设备创建，由注入调用方迁移设备。
        """
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = self.alpha / r
        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None
        # 标准 LoRA 初始化：A ~ Kaiming uniform，B = 0 → 初始增量为 0。
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=self.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=self.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算冻结基座输出与低秩增量之和。

        Args:
            x: 形状为 (..., in_features) 的输入，设备和精度应与权重兼容。

        Returns:
            形状为 (..., out_features) 的张量；梯度经增量分支传给 A/B。
        """
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """无梯度地返回 weight + (alpha/r) * B @ A，不修改本层。"""
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """将等效权重复制到新 nn.Linear，不原地替换本层。

        Returns:
            先以默认设备和精度创建的线性层；注入模块中的合并调用方再迁移它。
            本函数创建新权重会消耗 PyTorch 随机数，随后用合并值覆盖。
        """
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        lin.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            lin.bias.data.copy_(self.bias)
        return lin
