"""既有 QPiSSA 层：对冻结残差进行 NF4 量化。

打包数据、absmax 与码本保存在 CPU 普通属性中，不注册为 buffer，避免 .to()
改变打包数据的类型；前向时手动迁移并解量化。原浮点 residual buffer 仍保留。
量化实现改进属于第四阶段，本次仅整理文档。
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
    """对冻结残差做 NF4 量化的既有 PiSSA 层，保留普通精度 A/B。

    打包权重及量化元数据放在 CPU 普通属性中，不随 .to(...) 自动迁移。
    原 residual buffer 仍被保留；本次不改变其存储或序列化设计。
    """

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
        """创建普通 PiSSA 存储和未就绪的 CPU 量化属性。

        Args:
            in_features: 输入维度。
            out_features: 输出维度。
            r: 适配器秩。
            alpha: None 或 r，沿用父类限制。
            bias: 可选冻结偏置。
        """
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
        """从线性层执行 SVD、复制初始化结果，再量化残差。

        Args:
            linear: 源线性层，不原地修改。
            r: 适配器秩。
            alpha: None 或 r。
            niter: None 为精确 SVD，整数为 Fast SVD 幂迭代次数。

        Returns:
            新量化 PiSSA 层；沿用先默认设备和精度分配再复制的既有行为，
            注入调用方随后迁移设备和精度。此处不声称与源层逐位等价。
        """
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
        """将残差转为 CPU float32 做 NF4 量化，并缓存打包数据。

        已就绪时直接返回；原 residual buffer 仍保留，不在此处释放。
        量化数据、absmax 与码本作为普通属性保存，不自动进入 state_dict。
        """
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
        """在 A 的设备上重建 QuantState 并解量化，返回 A 精度的残差。

        量化尚未就绪时直接返回原 residual；就绪时每次调用都搬运并解量化。
        """
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
        """以解量化残差计算基座映射，再加低秩增量，输出形状 (..., out)。"""
        residual_fp = self._dequant_residual()
        base = F.linear(x, residual_fp, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """返回解量化残差加缩放 B @ A；结果通常不等于量化前源权重。"""
        return self._dequant_residual() + self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        """将解量化后的等效权重复制到默认设备和精度创建的新 nn.Linear。"""
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            linear.bias.data.copy_(self.bias)
        return linear
