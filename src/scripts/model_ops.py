"""模型加载、适配器调度、参数统计及优化器构建。"""
from __future__ import annotations

import torch
import torch.nn as nn

from utils import abspath, torch_dtype
from adapters.base import LowRankAdapterBase
from pissa import PiSSALinear, inject_pissa, merge_pissa, trainable_parameters
from pissa.inject_nf4 import inject_qpissa, merge_qpissa
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


def load_base_model(model_path: str, dtype: torch.dtype, trust_remote_code: bool = False, low_cpu_mem_usage: bool = True, quantization_config=None, device_map=None):
    """使用 Transformers 加载因果语言模型，不注入适配器。

    Args:
        model_path: 本地目录或 HF 仓库 ID；后者可能触发下载。
        dtype: 传给 from_pretrained 的模型精度。
        trust_remote_code: 是否允许执行模型仓库的自定义代码。
        low_cpu_mem_usage: 是否请求低 CPU 内存加载路径。
        quantization_config: 可选量化配置，原样传给加载器。
        device_map: 可选放置方式，原样传给加载器（量化加载时用 ``{"": 设备编号}`` 明确放到目标 GPU）。

    Returns:
        AutoModelForCausalLM 加载的模型；device_map 为 None 时不显式迁移至训练设备。
    """
    from transformers import AutoModelForCausalLM

    kwargs = {} if device_map is None else {"device_map": device_map}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code, quantization_config=quantization_config, **kwargs
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


def _set_trainable_from_adapters(model: nn.Module, expected_layers: int) -> None:
    """只放开真实适配器实例的 A/B，并核对可训练参数集合与适配器实例完全一致。

    Raises:
        RuntimeError: 适配器层数与预期不符，或可训练集合不等于全部 A/B。
    """
    adapters = [m for m in model.modules() if isinstance(m, LowRankAdapterBase)]
    if len(adapters) != expected_layers:
        raise RuntimeError(f"适配器实例数 {len(adapters)} 与替换层数 {expected_layers} 不一致")
    expected_ids = set()
    for m in adapters:
        for p in (m.lora_A, m.lora_B):
            p.requires_grad_(True)
            expected_ids.add(id(p))
    actual_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"可训练参数集合与适配器 A/B 不一致：多余 {len(actual_ids - expected_ids)}，缺少 {len(expected_ids - actual_ids)}"
        )


def _summarize_init_reports(reports: dict) -> dict:
    """汇总 QPiSSA 逐层初始化报告（训练基座与 Q_T 的偏差、重构误差、末轮量化误差）。"""
    if not reports:
        return {"num_layers": 0}
    dev = [r["requant_rel_dev_vs_Q_T"] for r in reports.values()]
    rec = [r["recon_rel_err_vs_W"] for r in reports.values()]
    last = [r["round_errors"][-1] for r in reports.values()]
    first = next(iter(reports.values()))
    return {
        "num_layers": len(reports),
        "rounds": first["rounds"],
        "requant_rel_dev_vs_Q_T": {"mean": sum(dev) / len(dev), "max": max(dev)},
        "recon_rel_err_vs_W": {"mean": sum(rec) / len(rec), "max": max(rec)},
        "final_round_quant_err": {"mean": sum(last) / len(last), "max": max(last)},
        "per_layer": reports,
    }


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
    extra: dict = {}
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
    elif method in ("qpissa", "qlora"):
        mcfg = cfg["methods"][method]
        qcfg = cfg["quantization"]
        adapter_dtype = torch_dtype(mcfg.get("adapter_dtype", "float32"))
        for p in model.parameters():
            p.requires_grad_(False)
        if method == "qpissa":
            # QPiSSA：浮点模型逐层做外层交替初始化，再按训练加载设置重新量化残差（HANDOFF P6/P7）。
            reports: dict = {}
            replaced = inject_qpissa(
                model, r=acfg["r"], alpha=acfg["alpha"], target_modules=tuple(acfg["target_modules"]),
                niter=acfg["niter"], rounds=acfg.get("init_rounds", 1),
                compute_dtype=torch_dtype(qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
                double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
                init_double_quant=qcfg.get("init_double_quant", False),
                adapter_dtype=adapter_dtype, base_dtype=torch_dtype(qcfg.get("train_load_dtype", "bfloat16")),
                device=next(model.parameters()).device, seed=acfg.get("init_seed"), reports=reports,
            )
            extra["qpissa_init"] = _summarize_init_reports(reports)
        else:
            # QLoRA：模型已按 4-bit 加载，直接给 Linear4bit 加 FP32（默认）适配器。
            replaced = inject_lora_4bit(
                model, r=acfg["r"], alpha=acfg["alpha"], target_modules=acfg["target_modules"],
                dtype=adapter_dtype,
            )
        coverage = check_replace_coverage(replaced, acfg["target_modules"])
        n_replaced = len(replaced)
        _set_trainable_from_adapters(model, len(replaced))
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
        **extra,
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
        merge_qpissa(model)  # 不再依赖 NFPiSSALinear 继承 PiSSALinear
    elif method == "lora":
        merge_lora(model)


MASTER_WEIGHT_MODES = ("none", "cpu_fp32")


def _available_host_bytes(cgroup_dir: str = "/sys/fs/cgroup", meminfo: str = "/proc/meminfo") -> int:
    """估计当前进程可用的主机内存（取 MemAvailable 与 cgroup 剩余额度的较小值）。

    cgroup 的 ``memory.current`` 含页缓存（读写大 checkpoint 后可达数十 GiB），而页缓存可被内核按需
    回收，因此剩余额度按 ``memory.max - (memory.current - active_file - inactive_file)`` 计算。
    """
    avail = None
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        limit = open(f"{cgroup_dir}/memory.max").read().strip()
        if limit != "max":
            used = int(open(f"{cgroup_dir}/memory.current").read().strip())
            reclaimable = 0
            try:
                for line in open(f"{cgroup_dir}/memory.stat"):
                    key, _, val = line.partition(" ")
                    if key in ("active_file", "inactive_file"):
                        reclaimable += int(val)
            except OSError:
                pass
            rest = int(limit) - max(used - reclaimable, 0)
            avail = rest if avail is None else min(avail, rest)
    except (OSError, ValueError):
        pass
    return avail if avail is not None else 1 << 62


class CPUMasterAdamW:
    """AdamW：FP32 主权重与一/二阶矩存放在主机内存，逐参数搬到 GPU 更新，再写回 BF16 权重。

    复现官方 DeepSpeed ZeRO-1/2 + bf16 的数值语义：前向/反向使用 BF16 参数与梯度，
    更新在 FP32 主权重上累积（不会因 BF16 舍入丢失小更新），再把主权重舍入回 BF16 参数。
    公式与 ``torch.optim.AdamW`` 一致（解耦权重衰减、偏差修正、amsgrad=False）。
    只用于 BF16 参数的 Full-FT；M3 量化后残差变小，不需要它。

    每个参数需要 12 字节主机内存（主权重 + m + v，均 FP32）；构造时会检查可用内存。
    可行时使用锁页内存并用独立 CUDA 流预取下一个参数的状态以重叠传输与计算。

    Attributes:
        param_groups: 与 torch 优化器兼容的单个参数组，训练循环通过其 ``lr`` 键调节学习率。
    """

    def __init__(self, params, lr: float, weight_decay: float, betas=(0.9, 0.999), eps: float = 1e-8) -> None:
        """从当前参数值初始化 FP32 主权重，m、v 置零。

        Raises:
            RuntimeError: 参数不在同一设备/不是 BF16 浮点，或主机内存不足。
        """
        self.params = [p for p in params]
        if not self.params:
            raise RuntimeError("没有可优化的参数")
        self.device = self.params[0].device
        if any(p.device != self.device for p in self.params):
            raise RuntimeError("CPUMasterAdamW 要求所有参数位于同一设备")
        if any(p.dtype != torch.bfloat16 for p in self.params):
            raise RuntimeError("CPUMasterAdamW 只用于 BF16 参数（FP32 参数本身就是主权重）")
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.param_groups = [{"params": self.params, "lr": lr, "weight_decay": weight_decay}]
        self.step_count = 0
        need = 12 * sum(p.numel() for p in self.params)
        avail = _available_host_bytes()
        if need > 0.95 * avail:
            raise RuntimeError(
                f"主机内存不足：需要约 {need / 2**30:.0f} GiB 存放 FP32 主权重与 Adam 状态，"
                f"当前可用约 {avail / 2**30:.0f} GiB。"
            )
        self.use_cuda = self.device.type == "cuda"
        self.pinned = self.use_cuda
        self.master, self.exp_avg, self.exp_avg_sq = [], [], []
        for p in self.params:
            self.master.append(self._host_like(p, init=p.detach().float()))
            self.exp_avg.append(self._host_like(p))
            self.exp_avg_sq.append(self._host_like(p))
        self.copy_stream = torch.cuda.Stream(self.device) if self.use_cuda else None
        print(f"  [CPUMasterAdamW] 主机内存占用约 {need / 2**30:.1f} GiB，锁页内存={self.pinned}")

    def _host_like(self, p: torch.Tensor, init: torch.Tensor | None = None) -> torch.Tensor:
        """在主机上分配与 p 同形状的 FP32 张量；锁页失败时退回普通内存。"""
        if self.pinned:
            try:
                t = torch.empty(p.shape, dtype=torch.float32, pin_memory=True)
            except RuntimeError:
                self.pinned = False
                print("  [warn] 锁页内存分配失败，改用普通主机内存（传输会变慢）")
                t = torch.empty(p.shape, dtype=torch.float32)
        else:
            t = torch.empty(p.shape, dtype=torch.float32)
        if init is None:
            t.zero_()
        else:
            t.copy_(init)
        return t

    def zero_grad(self, set_to_none: bool = True) -> None:
        """清除参数梯度（只支持 set_to_none=True 的语义）。"""
        for p in self.params:
            p.grad = None

    def _fetch(self, i: int):
        """把第 i 个参数的 FP32 状态搬到 GPU（在拷贝流上异步进行）。"""
        if not self.use_cuda:
            return self.master[i].clone(), self.exp_avg[i].clone(), self.exp_avg_sq[i].clone()
        with torch.cuda.stream(self.copy_stream):
            return tuple(t.to(self.device, non_blocking=True)
                         for t in (self.master[i], self.exp_avg[i], self.exp_avg_sq[i]))

    @torch.no_grad()
    def step(self) -> None:
        """执行一次 AdamW 更新；梯度为 None 的参数被跳过（与 torch 一致）。"""
        group = self.param_groups[0]
        lr, wd = group["lr"], group["weight_decay"]
        self.step_count += 1
        t = self.step_count
        bc1 = 1.0 - self.beta1 ** t
        bc2_sqrt = (1.0 - self.beta2 ** t) ** 0.5
        active = [i for i, p in enumerate(self.params) if p.grad is not None]
        main = torch.cuda.current_stream(self.device) if self.use_cuda else None
        nxt = self._fetch(active[0]) if active else None
        for pos, i in enumerate(active):
            p = self.params[i]
            cur = nxt
            if self.use_cuda:
                main.wait_stream(self.copy_stream)  # 状态已在 GPU 上就绪
                for x in cur:
                    x.record_stream(main)
            nxt = self._fetch(active[pos + 1]) if pos + 1 < len(active) else None
            w, m, v = cur
            g = p.grad.float()
            w.mul_(1.0 - lr * wd)
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            denom = (v.sqrt() / bc2_sqrt).add_(self.eps)
            w.addcdiv_(m, denom, value=-lr / bc1)
            p.data.copy_(w)  # 主权重舍入回 BF16 参数
            if self.use_cuda:
                self.copy_stream.wait_stream(main)
                with torch.cuda.stream(self.copy_stream):
                    self.master[i].copy_(w, non_blocking=True)
                    self.exp_avg[i].copy_(m, non_blocking=True)
                    self.exp_avg_sq[i].copy_(v, non_blocking=True)
            else:
                self.master[i].copy_(w)
                self.exp_avg[i].copy_(m)
                self.exp_avg_sq[i].copy_(v)
            del g, denom
        if self.use_cuda:
            torch.cuda.synchronize(self.device)  # 保证写回完成后再返回


OPTIMIZER_KINDS = ("adamw", "adamw_8bit")


def build_optimizer(params, lr: float, weight_decay: float, kind: str, master_weights: str = "none"):
    """构建 AdamW 或 8-bit AdamW；不再静默回退。

    Args:
        params: 交给优化器的参数迭代对象；调用方负责筛选可训练参数。
        lr: 学习率。
        weight_decay: 解耦权重衰减系数。
        kind: ``adamw``（torch.optim.AdamW，默认超参与官方 adamw_torch 一致）
            或 ``adamw_8bit``（bitsandbytes）。
        master_weights: ``none`` 或 ``cpu_fp32``（BF16 参数 + CPU 上的 FP32 主权重，仅 kind=adamw）。

    Returns:
        (优化器实例, 实际使用的优化器名称)。此处不执行参数更新。

    Raises:
        ValueError: kind 不在支持列表中。
        RuntimeError: 请求 adamw_8bit 但 bitsandbytes 不可用或构建失败；
            不回退到普通 AdamW，避免显存/数值条件被静默改变。
    """
    if kind not in OPTIMIZER_KINDS:
        raise ValueError(f"未知 optimizer={kind!r}，支持 {OPTIMIZER_KINDS}")
    if master_weights not in MASTER_WEIGHT_MODES:
        raise ValueError(f"未知 master_weights={master_weights!r}，支持 {MASTER_WEIGHT_MODES}")
    if master_weights == "cpu_fp32":
        if kind != "adamw":
            raise ValueError("cpu_fp32 主权重只支持 optimizer=adamw")
        return CPUMasterAdamW(params, lr=lr, weight_decay=weight_decay), "adamw_cpu_fp32_master"
    if kind == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay), "adamw_8bit"
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"请求 adamw_8bit 但 bitsandbytes 不可用或构建失败：{exc}") from exc
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay), "adamw"
