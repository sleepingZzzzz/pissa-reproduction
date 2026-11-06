"""PiSSA 的 SVD 初始化（从零实现，项目核心）。

约定（PyTorch nn.Linear 视角）：
    nn.Linear 的权重 ``weight`` 形状为 (out_features, in_features)，
    前向为 ``y = x @ weight.T``。

PiSSA 对该权重做经济型 SVD ``W = U @ diag(S) @ Vh``，取前 r 个主成分构造
可训练适配器 A、B，其余作为冻结的残差 ``W_res``：

    A (lora_A) = diag(sqrt(S[:r])) @ Vh[:r, :]     形状 (r, in)
    B (lora_B) = U[:, :r] @ diag(sqrt(S[:r]))       形状 (out, r)
    W_res      = W - B @ A                           形状 (out, in)

显然 ``B @ A`` 恰为 W 的主成分部分，故 ``W_res + B @ A == W``，
初始化不改变原模型输出（与论文 Eq.5 一致）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PiSSAInit:
    """一次初始化的 A、B、残差及奇异值，均采用源权重精度和设备。

    Attributes:
        lora_A: 形状 (r, in_features) 的主成分右因子。
        lora_B: 形状 (out_features, r) 的主成分左因子。
        residual: 形状 (out_features, in_features) 的剩余权重。
        singular_values: 精确 SVD 的完整谱，或 Fast SVD 的前 q 个近似值。
    """

    lora_A: torch.Tensor  # (r, in_features)
    lora_B: torch.Tensor  # (out_features, r)
    residual: torch.Tensor  # (out_features, in_features)，冻结
    # 奇异值（降序），用于画图/分析。注意长度取决于 niter：
    #   niter=None（精确 SVD）：全部 min(out, in) 个奇异值；
    #   Fast SVD：只有前 q 个近似值（默认 min(min(out, in), r+8)），不是完整谱。
    singular_values: torch.Tensor


def pissa_init(
    weight: torch.Tensor,
    r: int,
    niter: int | None = None,
    *,
    q: int | None = None,
) -> PiSSAInit:
    """对单个二维线性权重分离主成分低秩因子与残差。

    Args:
        weight: 形状 (out_features, in_features) 的权重，不原地修改。
        r: 主成分秩，满足 0 < r <= min(out_features, in_features)。
        niter: None 使用精确 SVD；整数使用随机低秩 SVD，表示幂迭代次数。
        q: Fast SVD 求解维度，满足 r <= q <= min(out_features, in_features)；
            None 保持 min(min(out_features, in_features), r+8) 的默认值。

    Returns:
        PiSSAInit。先脱离梯度并以 float32 计算，再转回源精度；
        A = sqrt(Sr) Vhr，B = Ur sqrt(Sr)，residual = W - B @ A。
        精确分支的谱长为 min(out, in)，Fast 分支为 q；Fast 消耗全局随机数。

    Raises:
        ValueError: 权重非二维、秩越界，或 q 不符合 Fast 分支的约束。
    """
    if weight.ndim != 2:
        raise ValueError(f"weight 必须是二维矩阵，收到 shape={tuple(weight.shape)}")
    out_features, in_features = weight.shape
    k = min(out_features, in_features)
    if not 0 < r <= k:
        raise ValueError(f"r 必须落在 (0, {k}]，收到 r={r}")
    if q is not None:
        if niter is None:
            raise ValueError("q 仅适用于 niter 非 None 的 Fast SVD")
        if isinstance(q, bool) or not isinstance(q, int) or not r <= q <= k:
            raise ValueError(f"q 必须是 [{r}, {k}] 内的整数，收到 q={q!r}")

    # SVD 在 float32/float64 上数值更稳，算完再转回原 dtype。
    orig_dtype = weight.dtype
    W = weight.detach().to(torch.float32)

    if niter is None:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
        singular_values = S
    else:
        # 随机化低秩 SVD：只求前 q 个，q 略大于 r 以提升精度。
        q = min(k, r + 8) if q is None else q
        U, S, V = torch.svd_lowrank(W, q=q, niter=niter)
        Ur, Sr, Vhr = U[:, :r], S[:r], V[:, :r].mH
        singular_values = S

    sqrt_S = torch.sqrt(Sr)
    lora_A = torch.diag(sqrt_S) @ Vhr  # (r, in)
    lora_B = Ur * sqrt_S.unsqueeze(0)  # (out, r)，等价于 Ur @ diag(sqrt_S)
    residual = W - lora_B @ lora_A

    return PiSSAInit(
        lora_A=lora_A.to(orig_dtype),
        lora_B=lora_B.to(orig_dtype),
        residual=residual.to(orig_dtype),
        singular_values=singular_values.to(orig_dtype),
    )


def reconstruction_error_against(init: PiSSAInit, weight: torch.Tensor) -> float:
    """返回重构 ``W_res + B@A`` 相对原始 ``weight`` 的相对 Frobenius 误差。"""
    recon = init.residual.float() + init.lora_B.float() @ init.lora_A.float()
    W = weight.detach().float()
    denom = torch.linalg.norm(W).clamp_min(1e-12)
    return float(torch.linalg.norm(recon - W) / denom)
