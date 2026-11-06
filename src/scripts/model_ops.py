"""模型加载、适配器调度、参数统计及优化器构建。"""
from __future__ import annotations

import torch
import torch.nn as nn

from utils import abspath
from pissa import PiSSALinear, inject_pissa, merge_pissa, trainable_parameters
from pissa.inject_nf4 import inject_qpissa
from lora import LoRALinear, inject_lora, merge_lora
from lora.inject_nf4 import inject_lora_4bit, unwrap_qlora


def resolve_model_path(cfg: dict) -> str:
    """按本地目录、ModelScope、HF 的优先顺序解析模型来源。

    Args:
        cfg: 含 model 配置块的字典；非空 local_path 优先于 source。

    Returns:
        可传给 from_pretrained 的本地目录字符串或 HF 仓库 ID。
        ModelScope 分支可能创建缓存目录并下载文件；HF 分支只返回 ID。

    Raises:
        ValueError: 未指定本地目录且 source 不受支持。
    """
    mcfg = cfg["model"]
    # local_path 优先级最高：直接指向已下载的本地目录，跳过 snapshot_download
    if mcfg.get("local_path", ""):
        return str(abspath(mcfg["local_path"]))
    if mcfg["source"] == "modelscope":
        from modelscope import snapshot_download

        cache_dir = abspath(mcfg["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[modelscope] 下载/定位模型 {mcfg['ms_repo']} ...")
        # 只取 safetensors 权重，跳过冗余的 .bin/.pth 双格式与无关大文件，
        # 避免误判"文件缺失"而重复下载 ~25GB 的 pytorch_model-*.bin。
        return snapshot_download(
            mcfg["ms_repo"],
            cache_dir=str(cache_dir),
            ignore_patterns=["*.bin", "*.pth", "*.pdf"],
        )
    if mcfg["source"] == "hf":
        return mcfg["hf_repo"]
    raise ValueError(f"未知 model.source={mcfg['source']}")


def load_tokenizer(model_path: str, trust_remote_code: bool = False):
    """加载分词器，缺少 pad token 时复用 EOS，不修改其填充方向。

    Args:
        model_path: 本地目录或 HF 仓库 ID；后者可能触发下载。
        trust_remote_code: 是否允许执行模型仓库的自定义代码。

    Returns:
        AutoTokenizer 加载的分词器。
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 不强制 padding_side：Collator 自行构造右填充张量，不依赖 tokenizer 的 padding 方向。
    # ChatGLM 等模型 tokenizer 的 _pad 断言 padding_side=="left"，强制改右会在 tokenize_example 中断言失败。
    return tok


def load_base_model(model_path: str, dtype: torch.dtype, trust_remote_code: bool = False, low_cpu_mem_usage: bool = True, quantization_config=None):
    """使用 Transformers 加载因果语言模型，不注入适配器。

    Args:
        model_path: 本地目录或 HF 仓库 ID；后者可能触发下载。
        dtype: 传给 from_pretrained 的模型精度。
        trust_remote_code: 是否允许执行模型仓库的自定义代码。
        low_cpu_mem_usage: 是否请求低 CPU 内存加载路径。
        quantization_config: 可选量化配置，原样传给加载器。

    Returns:
        AutoModelForCausalLM 加载的模型；本函数不显式将其迁移至训练设备。
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code, quantization_config=quantization_config
    )
    return model


def fix_unregistered_meta_tensors(model: nn.Module, device: torch.device) -> None:
    """为具有 meta 旋转缓存的子模块在目标设备重建频率、余弦和正弦缓存。

    Args:
        model: 要遍历的模型；匹配的子模块属性会被原地覆盖。
        device: 重建缓存的目标设备。

    只处理 cos_cached、sin_cached 均为张量且至少一个位于 meta 的模块。
    按 max_seq_len_cached、base 和 cos 最后一维推导缓存；缺失时使用
    2048、10000 和回退维度 128。inv_freq 形状为 (ceil(dim/2),)，
    cos/sin 为 (1, 1, max_len, 2*ceil(dim/2))，精度为 float32。
    本函数针对这类缓存布局，不是任意 RoPE 实现的通用修复。
    """
    for module in model.modules():
        cos = getattr(module, 'cos_cached', None)
        sin = getattr(module, 'sin_cached', None)
        if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
            continue
        if str(cos.device) != 'meta' and str(sin.device) != 'meta':
            continue

        max_len = getattr(module, 'max_seq_len_cached', 2048)
        base = getattr(module, 'base', 10000.0)
        # 从 cos_cached 的最后一维反推 dim（emb 形状 [1, 1, max_len, dim]）
        dim = cos.shape[-1] if cos.ndim >= 2 else 128

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        module.inv_freq = inv_freq
        t = torch.arange(max_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        module.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
        module.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)


def count_base_params(model: nn.Module) -> int:
    """在适配器注入之前统计基座参数元素数，作为占比分母。

    Args:
        model: 尚未将冻结权重转换为 buffer 的基座模型。

    Returns:
        Parameter 元素总数；量化参数有 quant_state.shape 时按还原形状统计。
        不计 buffer，因此不应在普通 PiSSA/LoRA 注入后计算该分母。
    """
    total = 0
    for p in model.parameters():
        qs = getattr(p, "quant_state", None)
        total += qs.shape.numel() if qs is not None and hasattr(qs, "shape") else p.numel()
    return total


def check_replace_coverage(replaced: list[str], targets) -> dict[str, int]:
    """统计每个目标叶名称的替换次数，缺失部分目标时打印警告。

    Args:
        replaced: 实际替换的完整模块名称列表。
        targets: 配置中的目标叶名称序列。

    Returns:
        目标名称到命中数的字典，供指标记录使用。

    Raises:
        RuntimeError: replaced 为空，即没有任何层被替换。
    """
    if not replaced:
        raise RuntimeError(f"没有任何层被替换，请检查 adapter.target_modules={list(targets)}")
    counts = {t: 0 for t in targets}
    for name in replaced:
        leaf = name.rsplit(".", 1)[-1]
        if leaf in counts:
            counts[leaf] += 1
    missing = [t for t, c in counts.items() if c == 0]
    if missing:
        print(f"  [warn] 以下 target_modules 一层都没命中：{missing}")
    return counts


def apply_adapter(model: nn.Module, method: str, cfg: dict) -> dict:
    """原地设置训练范围，按方法注入适配器并返回统计。

    Args:
        model: 基座模型；full 放开所有 Parameter，其余方法只放开 A/B。
        method: full、pissa、lora、qpissa 或 qlora。
        cfg: 含 adapter 的配置，提供秩、缩放、目标层和 SVD 设置。

    Returns:
        方法、替换层数、按目标名称的覆盖数、可训练元素数、注入前基座
        参数数及百分比组成的字典。total_params 不含新增适配器元素。

    Raises:
        ValueError: 方法名不受支持。
        RuntimeError: 适配器分支没有替换任何目标层。
    """
    acfg = cfg["adapter"]
    base_total = count_base_params(model)  # 必须在注入之前取（见 count_base_params）
    coverage: dict[str, int] = {}
    if method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        n_replaced = 0
    elif method == "pissa":
        for p in model.parameters():
            p.requires_grad_(False)
        replaced = inject_pissa(
            model,
            r=acfg["r"],
            alpha=acfg["alpha"],
            target_modules=tuple(acfg["target_modules"]),
            niter=acfg["niter"],
        )
        # 注入后只放开适配器参数。
        for n, p in model.named_parameters():
            p.requires_grad_("lora_A" in n or "lora_B" in n)
        n_replaced = len(replaced)
        coverage = check_replace_coverage(replaced, acfg["target_modules"])
    elif method == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        replaced = inject_lora(
            model, r=acfg["r"], alpha=acfg["alpha"], target_modules=acfg["target_modules"]
        )
        for n, p in model.named_parameters():
            p.requires_grad_("lora_A" in n or "lora_B" in n)
        n_replaced = len(replaced)
        coverage = check_replace_coverage(replaced, acfg["target_modules"])
    elif method == "qpissa":
        # QPiSSA: FP16 加载 → PiSSA SVD → NF4 量化残差
        for p in model.parameters():
            p.requires_grad_(False)
        replaced = inject_qpissa(
            model, r=acfg["r"], alpha=acfg["alpha"],
            target_modules=tuple(acfg["target_modules"]), niter=acfg["niter"],
        )
        for n, p in model.named_parameters():
            p.requires_grad_("lora_A" in n or "lora_B" in n)
        n_replaced = len(replaced)
        coverage = check_replace_coverage(replaced, acfg["target_modules"])
    elif method == "qlora":
        # QLoRA: 4-bit 模型上直接添加 LoRA（不 clone 4-bit weight）
        for p in model.parameters():
            p.requires_grad_(False)
        replaced = inject_lora_4bit(
            model, r=acfg["r"], alpha=acfg["alpha"], target_modules=acfg["target_modules"]
        )
        for n, p in model.named_parameters():
            p.requires_grad_("lora_A" in n or "lora_B" in n)
        n_replaced = len(replaced)
        coverage = check_replace_coverage(replaced, acfg["target_modules"])
    else:
        raise ValueError(f"未知 method={method}")

    trainable, _ = trainable_parameters(model)  # 第二个返回值只含 Parameter，口径不对，不用
    return {
        "method": method,
        "replaced_modules": n_replaced,
        "replaced_by_target": coverage,
        "trainable_params": trainable,
        # 分母 = 注入前的基座参数总量（论文口径，例如 LLaMA-2-7B 为 6,738,415,616）。
        "total_params": base_total,
        "trainable_pct": 100.0 * trainable / max(base_total, 1),
    }


def merge_adapter(model: nn.Module, method: str) -> None:
    """按方法原地合并 PiSSA、QPiSSA 或 LoRA，其余方法不执行操作。

    Args:
        model: 已注入适配器的模型。
        method: pissa/qpissa 调用 merge_pissa，lora 调用 merge_lora。

    本函数丢弃底层合并器返回的名称列表，始终返回 None；QLoRA 的导出由训练入口处理。
    """
    if method == "pissa":
        merge_pissa(model)
    elif method == "qpissa":
        merge_pissa(model)  # NFPiSSALinear is a subclass of PiSSALinear
    elif method == "lora":
        merge_lora(model)


def build_optimizer(params, lr: float, weight_decay: float, kind: str):
    """构建 AdamW；请求 8-bit 且导入或构建失败时打印警告并回退。

    Args:
        params: 交给优化器的参数迭代对象；调用方负责筛选可训练参数。
        lr: 学习率。
        weight_decay: 解耦权重衰减系数。
        kind: adamw_8bit 请求 bitsandbytes；其他值沿用普通 AdamW 分支。

    Returns:
        (优化器实例, 实际使用的优化器名称)。此处不执行参数更新。
    """
    if kind == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay), "adamw_8bit"
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] bitsandbytes 不可用（{exc}），回退普通 AdamW。")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay), "adamw"
