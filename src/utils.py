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
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件顶层必须是字典，收到 {type(cfg).__name__}：{path}")
    return cfg


def apply_overrides(cfg: dict[str, Any], assignments: list[str] | None) -> dict[str, Any]:
    """按 ``a.b.c=值`` 形式就地覆盖配置字段，值按 YAML 字面量解析。

    Args:
        cfg: 已加载的配置字典；被原地修改。
        assignments: 形如 ``methods.pissa.dtype=bfloat16`` 的字符串列表。

    Returns:
        同一个 cfg。路径中间节点必须已存在且为字典，叶子字段必须已存在，
        避免拼写错误悄悄新增字段。

    Raises:
        ValueError: 格式错误或路径不存在。
    """
    for item in assignments or []:
        if "=" not in item:
            raise ValueError(f"覆盖项必须形如 key=value，收到 {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise ValueError(f"覆盖路径不存在：{key}")
            node = node[p]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ValueError(f"覆盖路径不存在：{key}")
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


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
