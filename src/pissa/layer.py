"""PiSSALinear：替换 nn.Linear 的 PiSSA 适配器层（从零实现）。

前向：
    y = x @ W_res.T + bias + scaling * (x @ A.T) @ B.T
其中 W_res 冻结，A、B 可训练，scaling = alpha / r（PiSSA 推荐 alpha = r → scaling = 1）。
初始化时 W_res + B @ A == W，故输出与原始线性层一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .svd_init import pissa_init


class PiSSALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = self.alpha / r

        # 冻结的残差权重，注册为 buffer（不参与梯度，但随模型保存/搬运设备）。
        self.register_buffer("residual", torch.zeros(out_features, in_features))
        # 可训练的主成分适配器。
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        if bias is not None:
            # 原始 bias 保留并冻结（论文只微调 W 的主成分）。
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int,
        alpha: float | None = None,
        niter: int | None = None,
    ) -> "PiSSALinear":
        """由一个已有 nn.Linear 构造 PiSSALinear，并完成 PiSSA 初始化。"""
        init = pissa_init(linear.weight.data, r=r, niter=niter)
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            bias=linear.bias.data if linear.bias is not None else None,
        )
        with torch.no_grad():
            module.residual.copy_(init.residual)
            module.lora_A.copy_(init.lora_A)
            module.lora_B.copy_(init.lora_B)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.residual, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """返回合并后的等效权重 ``W_res + scaling * B @ A``，用于部署/导回。"""
        return self.residual + self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """把训练后的 PiSSALinear 折叠回标准 nn.Linear。"""
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"
        )
