"""PiSSA 复现：从零实现的核心包。

对外暴露：
    - pissa_init / PiSSAInit：单矩阵 SVD 初始化
    - PiSSALinear：替换 nn.Linear 的适配器层
    - inject_pissa / merge_pissa：模型级注入与折叠
"""

from .svd_init import (
    PiSSAInit,
    pissa_init,
    reconstruction_error_against,
)
from .layer import PiSSALinear
from .inject import (
    DEFAULT_TARGET_MODULES,
    inject_pissa,
    merge_pissa,
    trainable_parameters,
)

__all__ = [
    "PiSSAInit",
    "pissa_init",
    "reconstruction_error_against",
    "PiSSALinear",
    "DEFAULT_TARGET_MODULES",
    "inject_pissa",
    "merge_pissa",
    "trainable_parameters",
]
