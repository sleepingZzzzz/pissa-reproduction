"""[M3] PiSSALinear with NF4-quantized frozen residual (QPiSSA).

关键设计：NF4 量化数据（residual_nf4/residual_absmax/residual_code）在 **CPU 上
计算并存储为纯 Python 属性**，不注册为 nn.Buffer。forward 时手动搬到 GPU 解量化。
这样 model.to(device, dtype) 不会破坏 NF4 打包数据。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.functional import QuantState

from .layer import PiSSALinear
from .svd_init import pissa_init


class NFPiSSALinear(PiSSALinear):
    """PiSSALinear with NF4-quantized frozen residual."""

    _nf4_ready: bool
    residual_nf4: torch.Tensor      # CPU uint8 packed NF4 weights
    residual_absmax: torch.Tensor   # CPU float
    residual_code: torch.Tensor     # CPU float (codebook)
    _nf4_shape_0: int
    _nf4_shape_1: int
    _nf4_dtype: str

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__(in_features, out_features, r, alpha, bias)
        self._nf4_ready = False
        self.residual_nf4 = torch.empty(0, dtype=torch.uint8)
        self.residual_absmax = torch.empty(0)
        self.residual_code = torch.empty(0)
        self._nf4_shape_0 = 0
        self._nf4_shape_1 = 0
        self._nf4_dtype = ""

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int,
        alpha: float | None = None,
        niter: int | None = None,
    ) -> "NFPiSSALinear":
        """构造并量化残差。"""
        init = pissa_init(linear.weight.data, r=r, niter=niter)
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            bias=linear.bias.data.clone() if linear.bias is not None else None,
        )
        with torch.no_grad():
            module.lora_A.copy_(init.lora_A)
            module.lora_B.copy_(init.lora_B)
            module.residual.copy_(init.residual)
        del init

        module.quantize_residual()
        return module

    @torch.no_grad()
    def quantize_residual(self) -> None:
        """NF4 量化当前 residual（CPU 上执行），量化后释放 residual buffer。"""
        if self._nf4_ready:
            return
        # 搬到 CPU 做 NF4 量化
        W = self.residual.data.float().cpu()
        W_nf4, qs = bnb.functional.quantize_nf4(W)

        # 全部存为纯 Python 属性（CPU），不注册 buffer → model.to() 不会碰
        self.residual_nf4 = W_nf4         # CPU, uint8
        self.residual_absmax = qs.absmax  # CPU
        self.residual_code = qs.code      # CPU
        self._nf4_shape_0 = qs.shape[0]
        self._nf4_shape_1 = qs.shape[1]
        self._nf4_dtype = str(self.residual.dtype)
        self._nf4_ready = True

        # 注意：不释放 self.residual buffer。它作为 registered buffer 参与
        # state_dict / model.to() 等流程；强行改 shape 会破坏 PyTorch 内部状态。

    def _dequant_residual(self) -> torch.Tensor:
        if not self._nf4_ready:
            return self.residual
        device = self.lora_A.device
        qs = QuantState(
            absmax=self.residual_absmax.to(device),
            shape=torch.Size([self._nf4_shape_0, self._nf4_shape_1]),
            code=self.residual_code.to(device),
            blocksize=64,
            quant_type="nf4",
            dtype=getattr(torch, self._nf4_dtype.split(".")[-1]) if self._nf4_dtype else torch.float16,
            offset=None,
            state2=None,
        )
        nf4_dev = self.residual_nf4.to(device)
        residual_dq = bnb.functional.dequantize_nf4(nf4_dev, qs)
        return residual_dq.to(self.lora_A.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_fp = self._dequant_residual()
        base = F.linear(x, residual_fp, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self._dequant_residual() + self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear
