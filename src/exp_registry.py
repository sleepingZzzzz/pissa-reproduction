"""实验编号与产物目录命名：以 ``logs/`` 中的最大 n 加 1 作为全局实验号。

命名规则：``(fail)_exp{n}_{...}_{MMDD_HH}``。
- n 只由 logs 根目录分配，同一次启动的 ckpt / results / logs 共用同一个 n 与时间戳；
- 创建时带 ``(fail)_`` 前缀，对应部分完整成功后由 :func:`mark_ok` 去掉；
- 按 n 查找时带/不带前缀各找一次（:func:`find_by_n`），结果最多一个。
"""
from __future__ import annotations

import fcntl
import re
from datetime import datetime
from pathlib import Path

FAIL_PREFIX = "(fail)_"
_NAME_RE = re.compile(r"^(\(fail\)_)?exp(\d+)(?:_|$)")


def parse_name(name: str) -> tuple[int, bool] | None:
    """解析名字中的实验号。

    Args:
        name: 目录或文件名，不含父路径。

    Returns:
        ``(n, is_fail)``；不符合 ``[(fail)_]exp{n}_`` 开头时为 None。
    """
    m = _NAME_RE.match(name)
    if m is None:
        return None
    return int(m.group(2)), m.group(1) is not None


def max_n(root: Path) -> int:
    """扫描 root 下一级条目（文件与目录），返回已用的最大 n；没有则为 0。"""
    root = Path(root)
    if not root.exists():
        return 0
    ns = [p[0] for p in (parse_name(e.name) for e in root.iterdir()) if p is not None]
    return max(ns, default=0)


def stamp(now: datetime | None = None) -> str:
    """返回 ``MMDD_HH`` 时间戳。"""
    return (now or datetime.now()).strftime("%m%d_%H")


def alloc_logs_dir(logs_root: Path, tag: str, now: datetime | None = None) -> tuple[int, str, Path]:
    """在 logs 根下分配 ``max_n+1`` 并创建带 fail 前缀的日志目录。

    用文件锁串行化“取最大值 + 创建”，避免并发启动拿到同一个 n。

    Args:
        logs_root: logs 根目录，不存在则创建。
        tag: 名字中 ``exp{n}_`` 与时间戳之间的部分，如 ``m1_Llama-2-7b_smoke``。
        now: 时间，缺省取当前时间。

    Returns:
        ``(n, 时间戳, 日志目录路径)``。
    """
    logs_root = Path(logs_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    st = stamp(now)
    with open(logs_root / ".alloc.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        n = max_n(logs_root) + 1
        path = logs_root / f"{FAIL_PREFIX}exp{n}_{tag}_{st}"
        path.mkdir()
    return n, st, path


def category_dir(root: Path, n: int, st: str, fail: bool = True) -> Path:
    """返回某类产物目录路径 ``root/[(fail)_]exp{n}_{MMDD_HH}``（不创建）。"""
    return Path(root) / f"{FAIL_PREFIX if fail else ''}exp{n}_{st}"


def find_by_n(root: Path, n: int) -> Path | None:
    """在 root 一级条目中按 n 查找，带/不带 fail 前缀都接受。

    Raises:
        RuntimeError: 同一个 n 匹配到多个条目（违反“一个 n 一次实验”）。
    """
    root = Path(root)
    if not root.exists():
        return None
    hits = [e for e in root.iterdir() if (parse_name(e.name) or (None,))[0] == n]
    if len(hits) > 1:
        raise RuntimeError(f"{root} 中 n={n} 匹配到多个条目：{[h.name for h in hits]}")
    return hits[0] if hits else None


def mark_ok(path: Path) -> Path:
    """去掉名字开头的 ``(fail)_`` 并返回新路径；本来就没有则原样返回。"""
    path = Path(path)
    if not path.name.startswith(FAIL_PREFIX):
        return path
    new = path.with_name(path.name[len(FAIL_PREFIX):])
    if new.exists():
        raise FileExistsError(f"目标已存在：{new}")
    path.rename(new)
    return new


def clear_versions(root: Path) -> list[str]:
    """删除 root 下所有符合 ``[(fail)_]exp{n}_`` 命名的目录（smoke 覆盖上次结果用）。

    只删名字匹配的一级目录，其他条目不动。

    Returns:
        被删除的目录名列表。
    """
    import shutil

    root = Path(root)
    removed = []
    if not root.exists():
        return removed
    for e in sorted(root.iterdir()):
        if e.is_dir() and parse_name(e.name) is not None:
            shutil.rmtree(e)
            removed.append(e.name)
    return removed
