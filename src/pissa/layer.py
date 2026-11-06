"""PiSSALinear：替换 nn.Linear 的 PiSSA 适配器层（从零实现）。

前向：
    y = x @ W_res.T + bias + scaling * (x @ A.T) @ B.T
其中 W_res 冻结，A、B 可训练，scaling = alpha / r（PiSSA 推荐 alpha = r → scaling = 1）。
初始化时 W_res + B @ A == W，故输出与原始线性层一致。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .svd_init import pissa_init


class PiSSALinear(nn.Module):
    """以冻结残差和可训练主成分低秩因子替换线性映射。

    residual 为 (out, in) buffer，A/B 为 (r, in)/(out, r) Parameter。
    偏置如存在则冻结；本实现要求 alpha == r，以保持初始化重构关系。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        """分配零残差、零 A/B 和可选冻结偏置，尚未执行 SVD 初始化。

        Args:
            in_features: 输入最后一维的长度。
            out_features: 输出最后一维的长度。
            r: 正整数适配器秩；与权重维度的关系由初始化函数检查。
            alpha: 缩放分子，None 时取 r；本实现要求等于 r。
            bias: 可选 (out_features,) 张量，克隆为 buffer。

        Raises:
            ValueError: r 非正整数，或 alpha 非有限正数、与 r 不等。
        """
        super().__init__()
        if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
            raise ValueError(f"r 必须是正整数，收到 r={r!r}")
        alpha_val = float(r) if alpha is None else float(alpha)
        if not math.isfinite(alpha_val) or alpha_val <= 0:
            raise ValueError(f"alpha 必须是有限的正数，收到 alpha={alpha!r}")
        if alpha_val != float(r):
            # 这是本实现的范围限制；一般 PiSSA 可补偿 scaling，但本复现不扩展该分支。
            # PiSSA 的初始化 W_res = W - B@A 只在 scaling = alpha / r = 1 时保证
            # W_res + scaling * B@A == W；alpha != r 会在初始化时改变模型输出。
            raise ValueError(
                f"PiSSA 要求 alpha == r（scaling = 1），收到 alpha={alpha!r}, r={r}；"
                "否则初始化会改变原模型输出。"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha_val
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
        """从源线性层做 SVD 初始化，保持其设备和精度。

        Args:
            linear: 源 nn.Linear，不原地修改。
            r: 主成分秩。
            alpha: None 或 r；其他值由构造器拒绝。
            niter: None 为精确 SVD，整数为随机低秩 SVD 幂迭代次数。

        Returns:
            新 PiSSALinear；只训练 A/B，残差和偏置冻结。数值精度内保持初始映射。
        """
        init = pissa_init(linear.weight.data, r=r, niter=niter)
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            bias=linear.bias.data if linear.bias is not None else None,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            module.residual.copy_(init.residual)
            module.lora_A.copy_(init.lora_A)
            module.lora_B.copy_(init.lora_B)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回残差映射与缩放低秩映射之和，输入 (..., in)，输出 (..., out)。"""
        base = F.linear(x, self.residual, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """返回合并后的等效权重 ``W_res + scaling * B @ A``，用于部署/导回。"""
        return self.residual + self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """将等效权重复制到默认设备、默认精度创建的新 nn.Linear。

        Returns:
            普通线性层。需要保留设备和精度时由调用方再执行 .to(...)。
            不原地修改本层；构造新线性层会消耗随机数。
        """
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear

    def extra_repr(self) -> str:
        """返回 repr 中的输入输出维度、秩及缩放信息。"""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"
        )
