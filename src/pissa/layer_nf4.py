"""QPiSSA 层：FP32 A/B + 由 bitsandbytes ``Linear4bit`` 管理的 NF4 冻结残差基座。

初始化协议（对齐官方 ``init_qpissa.py``，HANDOFF P6/P7）：

    阶段一（初始化，FP32，关闭 double quant）：外层交替迭代 ``rounds`` 轮
        Q_0 = 0
        B_t A_t = SVD_r(W - Q_{t-1});  R_t = W - B_t A_t;  Q_t = DQ(NF4(R_t))
    阶段二（训练加载，官方流程）：把 Q_T 转为 BF16，再以训练量化设置
        （默认 double quant 开启、compute dtype BF16）重新量化为 Linear4bit。
        因此训练基座一般不等于 Q_T，二者偏差记录在 ``init_report``。

``rounds=1`` 即只做一次 SVD 后量化残差的基线；外层轮数 ``rounds`` 与 Fast SVD 的
``niter``（幂迭代次数）是两个独立概念。完整浮点残差不被持久保存：量化就绪后，
本层只持有 Linear4bit 与 FP32 的 A/B。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from adapters.base import LowRankAdapterBase, validate_rank_alpha
from adapters.quant import build_linear4bit, dequantize_linear4bit, local_rng, nf4_roundtrip
from .svd_init import pissa_init


@dataclass
class QPiSSAInit:
    """交替初始化的结果（均为 FP32，位于输入权重所在设备）。

    Attributes:
        lora_A: ``(r, in)``，最后一轮 SVD 的右因子。
        lora_B: ``(out, r)``，最后一轮 SVD 的左因子。
        quantized_residual: ``Q_T = DQ(NF4(R_T))``，反量化后的 FP32 残差。
        round_errors: 每轮 ``||R_t - Q_t||_F / ||W||_F`` 与
            ``||W - Q_t - B_t A_t||_F / ||W||_F``（二者相等，均为该轮量化残差误差）。
    """

    lora_A: torch.Tensor
    lora_B: torch.Tensor
    quantized_residual: torch.Tensor
    round_errors: list[float] = field(default_factory=list)


@torch.no_grad()
def qpissa_alternating_init(
    weight: torch.Tensor,
    r: int,
    rounds: int = 1,
    niter: int | None = None,
    double_quant: bool = False,
    seed: int | None = None,
) -> QPiSSAInit:
    """对单个权重做 QPiSSA 外层交替初始化。

    Args:
        weight: ``(out, in)`` 权重；先转为 FP32 计算，不原地修改。
        r: 适配器秩。
        rounds: 外层交替轮数（>=1）；1 为一次 SVD 后量化残差的基线，论文配置为 5。
        niter: None 为精确 SVD；整数为 Fast SVD 的幂迭代次数（与 ``rounds`` 无关）。
        double_quant: 初始化阶段是否对 NF4 absmax 做嵌套量化；协议为 False。
        seed: 非 None 时用局部 RNG 固定 Fast SVD 的随机抽样，不影响全局 RNG。

    Returns:
        QPiSSAInit。后续轮的 A/B 不再是 W 的前 r 个主成分（它们是 ``W - Q_{t-1}`` 的主成分）。

    Raises:
        ValueError: ``rounds`` 不是正整数，或 r 越界。
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError(f"rounds 必须是 >=1 的整数，收到 {rounds!r}")
    W = weight.detach().to(torch.float32)
    w_norm = torch.linalg.norm(W).clamp_min(1e-30)
    Q = torch.zeros_like(W)
    A = B = None
    errors: list[float] = []
    for _ in range(rounds):
        with local_rng(seed, W.device):
            init = pissa_init(W - Q, r=r, niter=niter)
        A, B = init.lora_A, init.lora_B
        R = W - B @ A
        Q = nf4_roundtrip(R, compress_statistics=double_quant)
        errors.append(float(torch.linalg.norm(R - Q) / w_norm))
    return QPiSSAInit(lora_A=A, lora_B=B, quantized_residual=Q, round_errors=errors)


class NFPiSSALinear(LowRankAdapterBase):
    """QPiSSA 层：``y = base4bit(x) + scaling * B(A(x))``，base 为量化残差。

    本类不再继承 ``PiSSALinear``，也不持有完整浮点残差；合并请使用
    ``pissa.inject_nf4.merge_qpissa``。
    """

    init_report: dict

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int,
        alpha: float | None = None,
        *,
        rounds: int = 1,
        niter: int | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        init_double_quant: bool = False,
        adapter_dtype: torch.dtype = torch.float32,
        base_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> "NFPiSSALinear":
        """从浮点线性层做交替初始化并构造量化基座层，不修改源层。

        Args:
            linear: 未量化的浮点源 ``nn.Linear``。
            r: 适配器秩，须满足 ``0 < r <= min(in, out)``。
            alpha: None 或 r；PiSSA 要求 alpha == r。
            rounds: 外层交替轮数。
            niter: Fast SVD 幂迭代次数，None 为精确 SVD。
            compute_dtype: 训练阶段 Linear4bit 的计算精度。
            double_quant: 训练阶段重新量化时是否开启 double quant（官方默认开启）。
            init_double_quant: 初始化阶段是否开启 double quant（协议为 False）。
            adapter_dtype: A/B 存储精度（论文：FP32）。
            base_dtype: 把 ``Q_T`` 转成该精度后再重新量化（官方训练加载阶段为 BF16）。
            device: CUDA 设备，None 时取源权重设备。
            seed: Fast SVD 的局部随机种子。

        Returns:
            量化就绪的层；``init_report`` 记录每轮误差、重构误差及训练基座与 ``Q_T`` 的偏差。

        Raises:
            TypeError: 源层是量化层或权重非浮点。
            ValueError: 参数不合法。
        """
        if not isinstance(linear, nn.Linear) or not linear.weight.is_floating_point():
            raise TypeError(f"QPiSSA 需要未量化的浮点 nn.Linear，收到 {type(linear).__name__}")
        if {"Linear4bit", "Linear8bitLt"} & {c.__name__ for c in type(linear).__mro__}:
            raise TypeError("QPiSSA 不能对已量化层做初始化。")
        r, alpha_val = validate_rank_alpha(r, alpha)
        if alpha_val != float(r):
            raise ValueError(f"QPiSSA 要求 alpha == r（scaling = 1），收到 alpha={alpha!r}, r={r}")
        dev = torch.device(device) if device is not None else linear.weight.device

        W = linear.weight.detach().to(dev, torch.float32)
        init = qpissa_alternating_init(W, r, rounds=rounds, niter=niter, double_quant=init_double_quant, seed=seed)
        bias = linear.bias.detach() if linear.bias is not None else None
        base = build_linear4bit(
            init.quantized_residual.to(base_dtype), bias, compute_dtype, double_quant, dev,
        )
        layer = cls(base, linear.in_features, linear.out_features, r, alpha_val, dtype=adapter_dtype, device=dev)
        with torch.no_grad():
            layer.lora_A.copy_(init.lora_A)
            layer.lora_B.copy_(init.lora_B)
            q_train = dequantize_linear4bit(base).float()
            denom = torch.linalg.norm(init.quantized_residual).clamp_min(1e-30)
            requant_dev = float(torch.linalg.norm(q_train - init.quantized_residual) / denom)
            recon = q_train + layer.scaling * (layer.lora_B.float() @ layer.lora_A.float())
            recon_err = float(torch.linalg.norm(recon - W) / torch.linalg.norm(W).clamp_min(1e-30))
        layer.init_report = {
            "rounds": rounds,
            "round_errors": init.round_errors,
            "requant_rel_dev_vs_Q_T": requant_dev,
            "recon_rel_err_vs_W": recon_err,
        }
        return layer

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """返回 FP32 的 ``dequant(base) + scaling * B @ A``；通常不等于量化前源权重。"""
        return dequantize_linear4bit(self.base).float() + self.scaling * (
            self.lora_B.float() @ self.lora_A.float()
        )

    @torch.no_grad()
    def to_linear(self, dtype: torch.dtype = torch.float32) -> nn.Linear:
        """把等效权重复制到本层所在设备上新建的 ``nn.Linear``（默认 FP32）。

        在 meta 设备上构造再 ``to_empty``，不消耗任何随机数。
        """
        device = self.lora_A.device
        with torch.device("meta"):
            linear = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        linear = linear.to_empty(device=device)
        linear.weight.data.copy_(self.merged_weight().to(dtype))
        if self.base.bias is not None:
            linear.bias = nn.Parameter(self.base.bias.detach().to(device, dtype).clone())
        linear.weight.requires_grad_(False)
        return linear.to(dtype)
