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


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(model_path: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    return model


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
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
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
# 适配器注入：根据方法配置模型可训练性
# ---------------------------------------------------------------------------
def apply_adapter(model: nn.Module, method: str, cfg: dict) -> dict:
    """就地把模型改造为指定方法的可训练形态，返回参数统计。"""
    acfg = cfg["adapter"]
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
    elif method == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        replaced = inject_lora(
            model, r=acfg["r"], alpha=acfg["alpha"], target_modules=acfg["target_modules"]
        )
        for n, p in model.named_parameters():
            p.requires_grad_("lora_A" in n or "lora_B" in n)
        n_replaced = len(replaced)
    else:
        raise ValueError(f"未知 method={method}")

    trainable, total = trainable_parameters(model)
    return {
        "method": method,
        "replaced_modules": n_replaced,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def merge_adapter(model: nn.Module, method: str) -> None:
    if method == "pissa":
        merge_pissa(model)
    elif method == "lora":
        merge_lora(model)
    # full：无需合并。


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
