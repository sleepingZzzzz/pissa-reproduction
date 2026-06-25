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
    """一次 PiSSA 初始化的产物（均为 PyTorch nn.Linear 约定下的张量）。"""

    lora_A: torch.Tensor  # (r, in_features)
    lora_B: torch.Tensor  # (out_features, r)
    residual: torch.Tensor  # (out_features, in_features)，冻结
    singular_values: torch.Tensor  # 全部奇异值（降序），用于画图/分析


def pissa_init(
    weight: torch.Tensor,
    r: int,
    niter: int | None = None,
) -> PiSSAInit:
    """对单个权重矩阵做 PiSSA 初始化。

    参数
    ----
    weight: (out_features, in_features) 的二维张量。
    r:      主成分（适配器）秩。
    niter:  若为 None，用精确 SVD（``torch.linalg.svd``）；
            若为整数，用快速随机化 SVD（``torch.svd_lowrank``，对应论文的 Fast SVD），
            niter 为幂迭代次数。
    """
    if weight.ndim != 2:
        raise ValueError(f"weight 必须是二维矩阵，收到 shape={tuple(weight.shape)}")
    out_features, in_features = weight.shape
    k = min(out_features, in_features)
    if not 0 < r <= k:
        raise ValueError(f"r 必须落在 (0, {k}]，收到 r={r}")

    # SVD 在 float32/float64 上数值更稳，算完再转回原 dtype。
    orig_dtype = weight.dtype
    W = weight.detach().to(torch.float32)

    if niter is None:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
        singular_values = S
    else:
        # 随机化低秩 SVD：只求前 q 个，q 略大于 r 以提升精度。
        q = min(k, r + 8)
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
