"""[M1] 公共工具：配置加载、模型/分词器、适配器注入（PiSSA/LoRA/Full）、数据分词。

被 prepare_data.py / train.py / eval_gsm8k.py / plot_curves.py 复用，避免重复。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pissa import PiSSALinear, inject_pissa, merge_pissa, trainable_parameters  # noqa: E402

# MetaMath / Alpaca 风格提示模板（与 MetaMathQA 官方一致）。
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


# ---------------------------------------------------------------------------
# 配置 / 随机种子 / 路径
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def abspath(rel: str | Path) -> Path:
    """把配置中的相对路径解析到仓库根。"""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


# ---------------------------------------------------------------------------
# 模型下载 / 加载
# ---------------------------------------------------------------------------
def resolve_model_path(cfg: dict) -> str:
    """返回可直接喂给 from_pretrained 的本地目录或 repo id。"""
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
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 不强制 padding_side：Collator 自行构造右填充张量，不依赖 tokenizer 的 padding 方向。
    # ChatGLM 等模型 tokenizer 的 _pad 断言 padding_side=="left"，强制改右会在 tokenize_example 中断言失败。
    return tok


def load_base_model(model_path: str, dtype: torch.dtype, trust_remote_code: bool = False, low_cpu_mem_usage: bool = True, quantization_config=None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code, quantization_config=quantization_config
    )
    return model


def fix_unregistered_meta_tensors(model: nn.Module, device: torch.device) -> None:
    """修复从 from_pretrained 加载后残留在 meta device 的未注册缓存 tensor。

    Baichuan2 等模型的 RotaryEmbedding 将 cos_cached/sin_cached/inv_freq 作为
    普通 tensor 属性（非 nn.Parameter/Buffer 注册），model.to(device) 不会移动它们。
    low_cpu_mem_usage 的 _fast_init 会使其留在 meta device，load_state_dict 不包含
    它们 → 首次 forward 时 .to(device) 报 ``NotImplementedError: Cannot copy out of
    meta tensor; no data!``。

    本函数遍历所有子模块，在目标 device 上重建这些缓存。
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


# ---------------------------------------------------------------------------
# LoRA 基线（与 PiSSALinear 同构，仅初始化不同：A 高斯、B 零）
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float | None = None) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = self.alpha / r
        self.register_buffer("weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None
        # 标准 LoRA 初始化：A ~ Kaiming uniform，B = 0 → 初始增量为 0。
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=self.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=self.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        lin.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            lin.bias.data.copy_(self.bias)
        return lin


# ---------------------------------------------------------------------------
# QLoRA: 在 4-bit 量化模型上添加 LoRA 适配器（保持 4-bit base 层不改动）
# ---------------------------------------------------------------------------
class QLoRALinear(nn.Module):
    """包装 4-bit Linear 层 + LoRA A/B，forward = 4bit_base(x) + lora_delta。"""

    def __init__(self, linear_4bit: nn.Linear, r: int, alpha: float | None = None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = linear_4bit.in_features
        self.out_features = linear_4bit.out_features
        self.r = r
        self.alpha = float(alpha) if alpha is not None else float(r)
        self.scaling = self.alpha / r
        # 保持 4-bit base 层作为 frozen 子模块（不 clone weight）
        self.base = linear_4bit
        for p in self.base.parameters():
            p.requires_grad_(False)
        # LoRA A/B 使用计算精度
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        raise NotImplementedError("QLoRA 不能合并 4-bit base weight")

    @torch.no_grad()
    def to_linear(self) -> nn.Linear:
        raise NotImplementedError("QLoRA 不能折叠为 nn.Linear")


def _match(name: str, targets: tuple[str, ...]) -> bool:
    return name.rsplit(".", 1)[-1] in targets


def inject_lora(model: nn.Module, r: int, alpha, target_modules) -> list[str]:
    replaced: list[str] = []
    to_replace = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and _match(n, tuple(target_modules))
    ]
    for name, module in tqdm(to_replace, desc="LoRA 注入", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        layer = LoRALinear(module, r=r, alpha=alpha).to(module.weight.device, module.weight.dtype)
        setattr(parent, child, layer)
        replaced.append(name)
    return replaced


def merge_lora(model: nn.Module) -> list[str]:
    merged: list[str] = []
    to_merge = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, module in tqdm(to_merge, desc="合并 LoRA", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        lin = module.to_linear().to(module.lora_A.device, module.lora_A.dtype)
        setattr(parent, child, lin)
        merged.append(name)
    return merged


# ---------------------------------------------------------------------------
# QPiSSA (NF4-quantized residual) injection
# ---------------------------------------------------------------------------
def inject_qpissa(model: nn.Module, r: int, alpha, target_modules, niter=None) -> list[str]:
    from pissa.layer_nf4 import NFPiSSALinear

    replaced: list[str] = []
    to_replace = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and _match(n, tuple(target_modules))
    ]
    for name, module in tqdm(to_replace, desc="QPiSSA NF4 注入", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        layer = NFPiSSALinear.from_linear(module, r=r, alpha=alpha, niter=niter).to(
            module.weight.device, module.weight.dtype
        )
        setattr(parent, child, layer)
        del module  # 释放旧 Linear 层
        replaced.append(name)
        # 每 8 层清理一次 CUDA 缓存，防止 SVD 临时 tensor 累积
        if len(replaced) % 8 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    return replaced


# ---------------------------------------------------------------------------
# QLoRA (LoRA on top of 4-bit base)
# ---------------------------------------------------------------------------
def inject_lora_4bit(model: nn.Module, r: int, alpha, target_modules) -> list[str]:
    """在 4-bit 量化模型上添加 LoRA 适配器。保持 4-bit base 层不变。"""
    replaced: list[str] = []
    to_replace = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and _match(n, tuple(target_modules))
    ]
    # QLoRA 的基座必须已被 bitsandbytes 量化为 Linear4bit；匹配到普通 nn.Linear 说明
    # 模型没有按 4-bit 加载，此时静默包装会得到"未量化的 QLoRA"，直接报错。
    not_4bit = [n for n, m in to_replace if "Linear4bit" not in {c.__name__ for c in type(m).__mro__}]
    if not_4bit:
        raise TypeError(f"QLoRA 目标层不是 Linear4bit（模型未按 4-bit 加载？）：{not_4bit[:3]} …共 {len(not_4bit)} 层")
    for name, module in tqdm(to_replace, desc="QLoRA 注入", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        # 4-bit Linear4bit → QLoRALinear wrapper（不 clone uint8 weight）
        layer = QLoRALinear(module, r=r, alpha=alpha).to(module.weight.device)
        setattr(parent, child, layer)
        replaced.append(name)
    return replaced


def unwrap_qlora(model: nn.Module, adapter_sd: dict[str, torch.Tensor] | None = None) -> list[str]:
    """解包 QLoRALinear → 原始 Linear4bit，可选提取 adapter state_dict。"""
    unwrapped: list[str] = []
    to_unwrap = [(n, m) for n, m in model.named_modules() if isinstance(m, QLoRALinear)]
    for name, module in tqdm(to_unwrap, desc="解包 QLoRA", unit="layer"):
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        if adapter_sd is not None:
            adapter_sd[f"{name}.lora_A"] = module.lora_A.data.clone().cpu()
            adapter_sd[f"{name}.lora_B"] = module.lora_B.data.clone().cpu()
        setattr(parent, child, module.base)
        unwrapped.append(name)
    return unwrapped


# ---------------------------------------------------------------------------
# 适配器注入：根据方法配置模型可训练性
# ---------------------------------------------------------------------------
def count_base_params(model: nn.Module) -> int:
    """注入适配器之前的"基座参数总量"，用作可训练占比的分母。

    注入后冻结权重被注册成 buffer，model.parameters() 统计不到它们，
    所以分母必须在注入之前取。4-bit 量化参数（Params4bit）按还原后的元素数计。
    """
    total = 0
    for p in model.parameters():
        qs = getattr(p, "quant_state", None)
        total += qs.shape.numel() if qs is not None and hasattr(qs, "shape") else p.numel()
    return total


def check_replace_coverage(replaced: list[str], targets) -> dict[str, int]:
    """注入后检查覆盖：一层都没替换就报错；config 里的某个目标名没命中则告警。

    返回 {目标名: 命中层数}，写入 metrics，便于事后核对每个模型的实际覆盖范围。
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
    """就地把模型改造为指定方法的可训练形态，返回参数统计。"""
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
    if method == "pissa":
        merge_pissa(model)
    elif method == "qpissa":
        merge_pissa(model)  # NFPiSSALinear is a subclass of PiSSALinear
    elif method == "lora":
        merge_lora(model)
    # full：无需合并。
    # qlora：不合并 4-bit 基座，保存路径由 train.py 单独处理（unwrap_qlora），不应调用本函数。


# ---------------------------------------------------------------------------
# 数据：分词 + response-only loss 掩码
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_example(tokenizer, instruction: str, response: str, max_len: int, response_only: bool):
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    resp_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + resp_ids)[:max_len]
    if response_only:
        labels = ([-100] * len(prompt_ids) + resp_ids)[:max_len]
    else:
        labels = list(input_ids)
    return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def build_optimizer(params, lr: float, weight_decay: float, kind: str):
    """kind ∈ {adamw, adamw_8bit}。adamw_8bit 需要 bitsandbytes，缺失则回退。"""
    if kind == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay), "adamw_8bit"
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] bitsandbytes 不可用（{exc}），回退普通 AdamW。")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay), "adamw"
