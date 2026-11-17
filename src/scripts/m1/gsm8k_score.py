"""GSM8K 评分：官方提取/判定规则（主评分）与 Decimal 精确比较（副评分）。

主评分逐行对齐官方评测参考 f7509eee 的 ``inference/acc_gsm8k.py::extract_answer_number``
（HANDOFF P8）：取最后一个 ``The answer is: `` 之后的文本，用正则取第一个数字候选，
处理千位逗号与分数，``round(float(...))`` 取整，再与 ``float(gold)`` 比较；
没有标记或没有数字候选时返回 None（计为未答对），不做"取最后一个数字"的回退。

与官方的唯一实现差异：
    - 用标准库 ``fractions.Fraction`` 代替第三方 ``fraction`` 包（不新增依赖）；
    - 额外捕获 ValueError/OverflowError/ZeroDivisionError 并返回 None，避免一条异常输出中断整次评测。

副评分 ``decimal_v1`` 不取整：把同一候选字符串精确转为 Decimal/Fraction 后与 gold 比较。
两种评分在看到任何测试成绩之前就已固定，各方法统一应用，不据 GSM8K test 成绩选择规则。
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

SCORER_PRIMARY = "official_f7509eee_v1"
SCORER_SECONDARY = "decimal_v1"
ANSWER_MARKER = "The answer is: "
# 官方评测参考 f7509eee 的停止串；官方 main 的 gen_vllm.py 为空列表（变体）。
OFFICIAL_STOP_STRINGS_F7509EEE = [
    "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT",
    "Instruction:", "Instruction", "Response:", "Response",
]
_NUMBER = re.compile(r"[\-+]?\d*[\.,/]?\d+")


def is_number(s: str) -> bool:
    """官方 is_number：float() 可解析，或 unicodedata.numeric 可解析。"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def candidate_after_marker(completion: str) -> str | None:
    """返回最后一个答案标记之后第一个数字候选的原始字符串；没有则为 None。"""
    parts = completion.split(ANSWER_MARKER)
    if len(parts) <= 1:
        return None
    match = _NUMBER.search(parts[-1].strip())
    return match.group() if match else None


def extract_answer_number(completion: str):
    """官方提取：返回取整后的整数，或 None（无标记/无候选/无法解析/无穷）。"""
    cand = candidate_after_marker(completion)
    if cand is None:
        return None
    try:
        if "/" in cand:
            numerator, denominator = cand.split("/")[0], cand.split("/")[1]
            if is_number(denominator) and is_number(numerator):
                if denominator == "0":
                    return round(float(numerator.replace(",", "")))
                frac = Fraction(cand.replace(",", ""))
                return round(float(frac.numerator / frac.denominator))
            return None
        value = float(cand.replace(",", ""))
        if value == float("inf"):
            return None
        return round(value)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def gold_value(answer_field: str):
    """从 GSM8K answer 字段取最后一个 ``####`` 之后的数值（float）；无效返回 None。

    先剥离空白和千位逗号。没有 ``####`` 标记、后缀为空或无法转成有限浮点数均视为无效 gold，
    由预检统一排除，而不是计为模型未答对（E3）。
    """
    if "####" not in answer_field:
        return None
    raw = answer_field.split("####")[-1].strip().replace(",", "")
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


def score_primary(completion: str, gold: float) -> tuple[bool, int | None]:
    """官方判定：float(pred) == float(gold)；返回 (是否正确, 提取到的整数或 None)。"""
    pred = extract_answer_number(completion)
    if pred is None:
        return False, None
    return float(pred) == float(gold), pred


def score_secondary(completion: str, gold_text: str) -> tuple[bool, str | None]:
    """副评分 decimal_v1：同一候选不取整，精确数值比较；返回 (是否正确, 规范化候选字符串)。"""
    cand = candidate_after_marker(completion)
    if cand is None:
        return False, None
    try:
        raw_gold = gold_text.split("####")[-1].strip().replace(",", "")
        gold = Fraction(Decimal(raw_gold))
        c = cand.replace(",", "")
        if "/" in c:
            num, den = c.split("/")[0], c.split("/")[1]
            if Fraction(Decimal(den)) == 0:
                return False, c
            value = Fraction(Decimal(num)) / Fraction(Decimal(den))
        else:
            value = Fraction(Decimal(c))
        return value == gold, c
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return False, cand


def apply_stop(text: str, stops: list[str]) -> tuple[str, str | None]:
    """在最早出现的停止串处截断（停止串本身不保留），返回 (文本, 命中的停止串或 None)。"""
    best_idx, best = None, None
    for s in stops:
        idx = text.find(s)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx, best = idx, s
    if best_idx is None:
        return text, None
    return text[:best_idx], best
