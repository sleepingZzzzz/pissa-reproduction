"""跨实验复用的配置、随机种子、仓库路径与精度工具。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 YAML 配置，不合并默认配置或校验字段。

    Args:
        path: 配置文件路径；相对路径按当前工作目录解释。

    Returns:
        YAML 解析结果。调用方预期为字典；空文件实际返回 None。
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """设置 Python、NumPy、PyTorch CPU 和所有 CUDA 设备的全局随机种子。

    Args:
        seed: 随机种子；本函数不启用确定性算子，也不保证跨设备逐位一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def abspath(rel: str | Path) -> Path:
    """将相对路径接到仓库根，绝对路径保持原样。

    Args:
        rel: 配置中的路径；不要求路径存在，也不展开用户目录符号。

    Returns:
        仓库根下的 Path，或传入的绝对路径。
    """
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def torch_dtype(name: str) -> torch.dtype:
    """查表返回 float32、bfloat16 或 float16 对应的 torch.dtype。

    Args:
        name: 三种受支持的精度名称之一。

    Returns:
        对应的 PyTorch 精度对象。

    Raises:
        KeyError: 名称不在映射表中。
    """
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]
