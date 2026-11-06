"""M0：奇数类别预训练、偶数类别微调的 MNIST 机制实验，并非论文轨迹复刻。

每个 seed 训练一个共享基座，四种方法使用配对批次，测试集不用于挑选设置。
--scan 是训练集内部验证的单因素探索诊断：用独立划分种子留出 10% 偶数训练样本，
不加载官方测试集；主实验保留全部偶数训练样本及原随机流。
逐步位移针对有效 FC 权重，full_ft 也只统计其中 FC；位移本身不证明因果关系。
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import math
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

if __package__:
    from .verify_init import (ArgumentParser64, DEFAULT_OUTPUT, REPO_ROOT, atomic_json,
        close, finish_report, import_peft, new_report, relative, seeded)
else:
    from verify_init import (ArgumentParser64, DEFAULT_OUTPUT, REPO_ROOT, atomic_json,
        close, finish_report, import_peft, new_report, relative, seeded)
from pissa import PiSSALinear

DATA_DIR = REPO_ROOT / "data/mnist"
METHODS = ("fc_full", "full_ft", "lora", "pissa")
LABELS = {"fc_full": "FC weight+bias", "full_ft": "Full-model FT", "lora": "LoRA", "pissa": "PiSSA"}


class MLP(nn.Module):
    """三层线性网络：784 → hidden → hidden → 10，返回未归一化的分类 logits。"""
    def __init__(self, hidden=256):
        """创建编码层、中间 FC 和分类头；hidden 为两层隐藏宽度。"""
        super().__init__()
        self.enc = nn.Linear(784, hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 10)

    def forward(self, x):
        """将图像展平为 (N, 784)，经过两次 ReLU 后返回 (N, 10) logits。"""
        return self.head(F.relu(self.fc(F.relu(self.enc(x.flatten(1))))))


class LoRALinear(nn.Module):
    """M0 高斯初始化 LoRA：A 标准差为 1/r，B 为零，冻结基座和偏置。

    这是 M0 实验条件，不与 M1 的 Kaiming 初始化 LoRA 合并。
    """
    def __init__(self, linear, r, alpha=None):
        """校验秩与缩放，在源设备、源精度创建冻结基座和可训练 A/B。"""
        super().__init__()
        if isinstance(r, bool) or not isinstance(r, int) or not 0 < r <= min(linear.weight.shape):
            raise ValueError("r must be an integer in (0, min(weight.shape)]")
        alpha = r if alpha is None else float(alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.r, self.scaling = r, alpha/r
        self.register_buffer("weight", linear.weight.detach().clone())
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features,
            device=linear.weight.device, dtype=linear.weight.dtype)/r)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r,
            device=linear.weight.device, dtype=linear.weight.dtype))

    def forward(self, x):
        """返回基座输出与缩放低秩增量，输入 (..., in)，输出 (..., out)。"""
        return F.linear(x, self.weight, self.bias)+self.scaling*F.linear(F.linear(x, self.lora_A), self.lora_B)


def seed_bundle(seed):
    """由一个 seed 派生基座、预训练数据、适配器和微调数据的四路固定种子。"""
    return {"base": seed, "pretrain_data": seed+100_003,
            "adapter": seed+200_003, "finetune_data": seed+300_007}


def make_loader(dataset, batch, seed, shuffle):
    """使用独立的 CPU Generator 构建 DataLoader，不消耗全局随机流。"""
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=False)


def batch_stream(loader):
    """无限重复遍历非空 loader；若一轮没有任何批次则抛出 ValueError。"""
    if not len(loader):
        raise ValueError("empty training loader")
    while True:
        yield from loader


def get_loaders(batch, *, seed=0, train_dataset=None, test_dataset=None):
    """构建奇数训练、偶数微调及偶数测试三个数据加载器。

    Args:
        batch: 每批样本数。
        seed: 数据加载器的独立种子。
        train_dataset: 可选注入训练数据。
        test_dataset: 可选注入测试数据；两者任一为 None 时都重新加载官方 MNIST。

    Returns:
        odd、even、test 三个 loader 的字典；测试数据只用于预先约定的评测。
    """
    if train_dataset is None or test_dataset is None:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
        # Data cache is independent of --output. No test fixture reaches here.
        train_dataset = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)
    def subset(dataset, parity):
        """按 parity 指定的奇偶性筛选 MNIST targets，返回 Subset。"""
        labels = torch.as_tensor(dataset.targets)
        indices = (labels % 2 == parity).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            raise ValueError("empty odd/even dataset")
        return Subset(dataset, indices)
    return {"odd": make_loader(subset(train_dataset, 1), batch, seed, True),
            "even": make_loader(subset(train_dataset, 0), batch, seed, True),
            "test": make_loader(subset(test_dataset, 0), batch, seed, False)}


def get_scan_loaders(batch, *, seed=0, split_seed=20261101, validation_fraction=.1,
                     train_dataset=None):
    """仅加载训练数据，使用独立随机流划分偶数训练与验证样本。

    Args:
        batch: 每批样本数。
        seed: loader 种子。
        split_seed: 独立划分种子。
        validation_fraction: 偶数训练数据中留作验证的比例。
        train_dataset: 可选注入数据；None 时加载或下载 MNIST 训练集。

    Returns:
        奇数预训练、偶数微调、验证 loader 及划分信息；不加载官方测试集。
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if train_dataset is None:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
        train_dataset = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
    labels = torch.as_tensor(train_dataset.targets)
    even = (labels % 2 == 0).nonzero(as_tuple=True)[0].tolist()
    odd = (labels % 2 == 1).nonzero(as_tuple=True)[0].tolist()
    count = int(len(even)*validation_fraction)
    if not odd or not 0 < count < len(even):
        raise ValueError("split must have nonempty odd, even-train and validation subsets")
    order = torch.randperm(len(even), generator=torch.Generator().manual_seed(split_seed)).tolist()
    validation_ids = sorted(even[i] for i in order[:count])
    training_ids = sorted(even[i] for i in order[count:])
    split = {"seed": split_seed, "fraction": validation_fraction,
        "training_samples": len(training_ids), "validation_samples": len(validation_ids),
        "validation_indices": validation_ids,
        "training_indices_sha256": hashlib.sha256(",".join(map(str, training_ids)).encode()).hexdigest(),
        "role": "even training holdout; official test never loaded"}
    return {"odd": make_loader(Subset(train_dataset, odd), batch, seed, True),
        "even": make_loader(Subset(train_dataset, training_ids), batch, seed, True),
        "validation": make_loader(Subset(train_dataset, validation_ids), batch, split_seed, False),
        "split": split}


@torch.no_grad()
def effective_fc_weight(model):
    """返回 FC 的实际等效权重，适配器分支包含缩放 B @ A，而非裸因子。"""
    if isinstance(model.fc, PiSSALinear):
        return model.fc.merged_weight().detach()
    if isinstance(model.fc, LoRALinear):
        return (model.fc.weight + model.fc.scaling*(model.fc.lora_B @ model.fc.lora_A)).detach()
    return model.fc.weight.detach()


def validate_initialization(base, model, method, r, probe):
    """检查训练范围、初始 logits 和损失与基座一致，返回初始化诊断。"""
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    expected = set(dict(base.named_parameters())) if method == "full_ft" else (
        {"fc.weight", "fc.bias"} if method == "fc_full" else {"fc.lora_A", "fc.lora_B"})
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_count = sum(p.numel() for p in base.parameters()) if method == "full_ft" else (
        base.fc.weight.numel()+base.fc.bias.numel() if method == "fc_full" else
        r*(base.fc.in_features+base.fc.out_features))
    if names != expected or count != expected_count:
        raise ValueError(f"incorrect trainability for {method}: {sorted(names)}, count={count}")
    with torch.no_grad():
        reference, actual = base(probe), model(probe)
        labels = torch.zeros(len(probe), dtype=torch.long, device=probe.device)
        loss_ref, loss_actual = F.cross_entropy(reference, labels), F.cross_entropy(actual, labels)
    if not close(actual, reference) or not close(loss_actual, loss_ref):
        raise ValueError(f"initial function changed for {method}")
    if method == "pissa" and not close(model.fc.merged_weight(), base.fc.weight):
        raise ValueError("PiSSA residual + BA does not reconstruct fc.weight")
    return {"trainable_names": sorted(names), "trainable_parameters": count,
            "initial_logits_close": True, "initial_loss_close": True,
            "initial_logits_rel_err": relative(actual, reference)}


def prepare_method(base, method, r, probe):
    """深拷贝基座并配置四种微调范围之一，返回模型及初始化检查统计。"""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    model = copy.deepcopy(base)
    for p in model.parameters():
        p.requires_grad_(method == "full_ft")
    if method == "fc_full":
        for p in model.fc.parameters():
            p.requires_grad_(True)
    elif method == "lora":
        model.fc = LoRALinear(model.fc, r)
    elif method == "pissa":
        model.fc = PiSSALinear.from_linear(model.fc, r, alpha=r)
    return model, validate_initialization(base, model, method, r, probe)


def frozen_state(model):
    """克隆冻结 Parameter 和全部 buffer，用于后续检查它们是否被改动。"""
    return {**{n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad},
            **{n: b.detach().clone() for n, b in model.named_buffers()}}


def validate_frozen(model, before):
    """逐项检查冻结快照与当前模型完全一致；不一致时抛出 ValueError。"""
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    if any(n not in current or not torch.equal(value, current[n]) for n, value in before.items()):
        raise ValueError("frozen parameter/buffer changed")


@torch.no_grad()
def evaluate(model, loader, device):
    """无梯度计算按样本加权的交叉熵和准确率，恢复调用前的训练模式。"""
    training = model.training
    model.eval()
    total_loss, correct, count = 0., 0, 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
    finally:
        model.train(training)
    if not count:
        raise ValueError("empty evaluation set")
    return {"loss": total_loss/count, "accuracy": correct/count, "samples": count,
            "role": "held-out official test; no selection"}


def pretrain(model, loader, epochs, device):
    """以 Adam 预训练奇数类别，返回每轮在线损失、更新步数与样本数。"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    losses, samples, steps = [], 0, 0
    for epoch in range(epochs):
        loss_sum, count = 0., 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite pretraining loss")
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())*len(y)
            count += len(y)
            steps += 1
        losses.append(loss_sum/count)
        samples += count
        print(f"pretrain epoch {epoch+1}/{epochs}: mean online loss={losses[-1]:.5f}", flush=True)
    return {"epoch_online_loss": losses, "optimizer_steps": steps, "samples_seen": samples}


def finetune(model, loader, steps, lr, device, eval_loader, eval_every, tail_window, *,
             record_displacement=True, evaluation_steps=None, evaluation_role=None):
    """以固定步数微调并定期评测，检查冻结状态，可选记录 FC 位移。

    Args:
        model: 已配置训练范围的模型，会原地更新。
        loader: 可循环的微调数据加载器。
        steps: 优化步数。
        lr: Adam 学习率。
        device: 训练设备。
        eval_loader: 定期评测的数据加载器。
        eval_every: 评测步数间隔。
        tail_window: 尾部在线损失平均所用的步数。
        record_displacement: 是否记录有效 FC 权重在相邻优化步之间的位移。
        evaluation_steps: 可选评测步集合；非 None 时替代固定间隔规则。
        evaluation_role: 可选评测用途标签，写入评测结果。

    Returns:
        损失曲线、评测点、冻结检查、计时和可选位移等统计。
        位移是描述性诊断，不能单独建立训练差异的因果解释。
    """
    before = frozen_state(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    stream, losses, evaluations, sizes = batch_stream(loader), [], [], []
    displacements = []
    previous = effective_fc_weight(model).clone() if record_displacement else None
    def evaluation(step):
        """返回给定 step 的评测结果，并按需附加用途标签。"""
        result = {"step": step, **evaluate(model, eval_loader, device)}
        if evaluation_role is not None:
            result["role"] = evaluation_role
        return result
    tick = time.perf_counter()
    evaluations.append(evaluation(0))
    eval_seconds = time.perf_counter()-tick
    started = time.perf_counter()
    model.train()
    for step in range(1, steps+1):
        x, y = next(stream)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite fine-tuning loss")
        loss.backward()
        opt.step()
        if record_displacement:
            current = effective_fc_weight(model)
            displacements.append(float(torch.linalg.vector_norm(current-previous)))
            previous = current.clone()
        losses.append(float(loss.detach()))
        sizes.append(len(y))
        should_evaluate = (step in evaluation_steps or step == steps) if evaluation_steps is not None else (
            step % eval_every == 0 or step == steps)
        if should_evaluate:
            tick = time.perf_counter()
            evaluations.append(evaluation(step))
            eval_seconds += time.perf_counter()-tick
    validate_frozen(model, before)
    n = min(tail_window, steps)
    return {"loss_curve": losses, "last_online_loss": losses[-1],
        "tail_online_mean": sum(a*b for a, b in zip(losses[-n:], sizes[-n:]))/sum(sizes[-n:]),
        "tail_window": n, "evaluations": evaluations, "frozen_unchanged": True,
        "samples_seen": sum(sizes), "evaluation_seconds": eval_seconds,
        "step_displacements": displacements,
        "displacement_scope": "effective fc.weight only; full_ft enc/head and all biases excluded",
        "displacement_metric": "Frobenius norm of post-update minus pre-update effective weight",
        "loop_seconds": time.perf_counter()-started,
        "loss_semantics": "online pre-update batch loss; tail mixes parameter states"}


def check_lora_peft(device="cpu"):
    """对齐 PEFT 的 M0 高斯初始化和非零因子前向；缺依赖时返回跳过原因。"""
    imports, skipped = import_peft()
    if skipped:
        return skipped
    LoraConfig, get_peft_model, *_ = imports
    with seeded(31, device):
        base = nn.Sequential(nn.Linear(16, 12)).to(device)
        ours = LoRALinear(base[0], 4)
        calls = []
        original_normal = nn.init.normal_
        def record_normal(tensor, mean=0., std=1., **kwargs):
            """记录 PEFT 调用 normal_ 的形状和分布参数，再执行原初始化。"""
            calls.append({"shape": list(tensor.shape), "mean": mean, "std": std})
            return original_normal(tensor, mean=mean, std=std, **kwargs)
        with patch.object(nn.init, "normal_", record_normal):
            adapted = get_peft_model(copy.deepcopy(base), LoraConfig(r=4, lora_alpha=4,
                target_modules=["0"], init_lora_weights="gaussian", lora_dropout=0., bias="none"))
        target = adapted.get_submodule("base_model.model.0")
        x = torch.randn(8, 16, device=device)
        with torch.no_grad():
            initial_ok = close(ours(x), base[0](x)) and close(target(x), base[0](x))
        gaussian_ok = any(c == {"shape": [4, 16], "mean": 0., "std": .25} for c in calls)
        zero_ok = not bool(torch.count_nonzero(ours.lora_B)) and not bool(torch.count_nonzero(target.lora_B["default"].weight))
        # Initialization distribution is checked separately (std=1/r). Copy
        # nonzero factors to check forward wiring; B=0 alone hides wiring bugs.
        with torch.no_grad():
            ours.lora_A.copy_(target.lora_A["default"].weight)
            ours.lora_B.normal_()
            target.lora_B["default"].weight.copy_(ours.lora_B)
        with torch.no_grad():
            actual, reference = ours(x), target(x)
        passed = initial_ok and gaussian_ok and zero_ok and close(actual, reference) and not ours.weight.requires_grad and not ours.bias.requires_grad
        passed = passed and not target.base_layer.weight.requires_grad and not target.base_layer.bias.requires_grad
    return {"passed": passed, "forward_rel_err": relative(actual, reference),
        "A_gaussian_std": .25, "B_initialization": "zero; nonzero matched factors used for forward test",
        "scaling": 1., "gaussian_calls": calls, "initial_function_preserved": initial_ok,
        "B_initially_zero": zero_ok, "peft_version": importlib.metadata.version("peft")}


def paired_summary(runs):
    """汇总同 seed 的 PiSSA 减 LoRA 测试损失差，仅做描述性统计。"""
    differences = []
    for run in runs:
        methods = run["methods"]
        differences.append({"seed": run["seed"], "pissa_minus_lora_test_loss":
            methods["pissa"]["evaluations"][-1]["loss"]-methods["lora"]["evaluations"][-1]["loss"]})
    values = [d["pissa_minus_lora_test_loss"] for d in differences]
    return {"paired_differences": differences, "pissa_lower_loss_count": sum(x<0 for x in values),
        "n": len(values), "descriptive_mean_difference": statistics.mean(values),
        "descriptive_sample_std": statistics.stdev(values) if len(values)>1 else None,
        "interpretation": "descriptive only; no statistical significance or universal ordering claim"}


def scan_grid(lrs, ranks, base_r, base_lr):
    """构建单因素学习率/秩扫描，复用公共基准单元，不生成完整交叉格。"""
    cells = {}
    for lr in lrs:
        for method in METHODS:
            cells[(method, lr, base_r)] = {"method": method, "lr": lr, "r": base_r, "axes": ["lr"]}
    for r in ranks:
        for method in ("lora", "pissa"):
            key = (method, base_lr, r)
            if key in cells:
                cells[key]["axes"].append("rank")
            else:
                cells[key] = {"method": method, "lr": base_lr, "r": r, "axes": ["rank"]}
    return list(cells.values())


def scan_protocol(args):
    """构造训练前落盘的扫描协议；不根据结果重写规则，也不访问测试集。"""
    return {"diagnostic_only": True, "origin": "exploratory follow-up to already observed main results",
        "evaluation_split": "training_even_validation_only", "official_test_loaded": False,
        "split_seed": args.split_seed, "validation_fraction": args.validation_fraction,
        "lrs": args.scan_lrs, "ranks": args.scan_ranks, "base_lr": args.lr, "base_r": args.r,
        "checkpoints": args.scan_checkpoints, "steps": args.ft_steps, "seeds": args.seeds,
        "paired_rule": "same seed: one pretrained base and identical fresh finetune generator per cell",
        "comparisons": ["PiSSA minus fc_full validation loss per seed at each lr/checkpoint",
            "PiSSA minus LoRA and fixed-lr fc_full reference per seed at each rank/checkpoint",
            "count negative paired differences; report complete cells, not only winners",
            "adjacent changes of mean signed gap as lr/rank increases"],
        "best_lr_rule": "per method minimize across-seed mean final validation loss; ties use smaller lr",
        "best_lr_reporting": "use that single selected lr for every seed; report paired gaps at both checkpoints",
        "interpretation": "same-lr and method-specific validation-selected best-lr comparisons answer different questions; selected validation scores are not independent test estimates",
        "causal_limit": "gap changes and effective-FC displacements identify candidate explanations, not proven mechanisms",
        "rank_limit": "full-rank adapters still have different parameterization/count from dense FC; no inference about paper Fig.2(a)"}


def scan_table(runs, checkpoints):
    """展开各 seed、实验单元及指定验证 checkpoint 的结果，返回表格行列表。"""
    rows = []
    for run in runs:
        for cell in run["experiments"]:
            result = cell["result"]
            for evaluation in result["evaluations"]:
                step = evaluation["step"]
                if step not in checkpoints:
                    continue
                movement = result["step_displacements"][:step]
                rows.append({"seed": run["seed"], "axes": "+".join(cell["axes"]),
                    "method": cell["method"], "lr": cell["lr"], "r": cell["r"], "step": step,
                    "validation_loss": evaluation["loss"], "validation_samples": evaluation["samples"],
                    "trainable_parameters": result["trainable_parameters"],
                    "fc_step_displacement": movement[-1],
                    "fc_mean_step_displacement": statistics.mean(movement)})
    return rows


def scan_comparisons(table, protocol):
    """按预定规则汇总配对差、相邻趋势和验证集选定学习率的比较。

    返回描述性比较；验证选定的分数不是独立测试估计，不作显著性或因果推断。
    """
    index = {(x["seed"], x["method"], x["lr"], x["r"], x["step"]): x for x in table}
    seeds, checkpoints = protocol["seeds"], protocol["checkpoints"]
    base_r, base_lr = protocol["base_r"], protocol["base_lr"]
    def paired(method_a, lr_a, r_a, method_b, lr_b, r_b, step):
        """在相同 seed 配对候选与参考的验证损失，返回差值及描述性汇总。"""
        differences = [{"seed": seed, "difference": index[(seed, method_a, lr_a, r_a, step)]["validation_loss"]-
                        index[(seed, method_b, lr_b, r_b, step)]["validation_loss"]} for seed in seeds]
        return {"paired_differences": differences, "a_lower_loss_count": sum(x["difference"]<0 for x in differences),
                "n": len(seeds), "mean_signed_gap": statistics.mean(x["difference"] for x in differences)}
    same_lr = [{"lr": lr, "step": step, **paired("pissa", lr, base_r, "fc_full", lr, base_r, step)}
               for lr in protocol["lrs"] for step in checkpoints]
    rank_rows = [{"r": r, "step": step,
        "pissa_minus_fc_full": paired("pissa", base_lr, r, "fc_full", base_lr, base_r, step),
        "pissa_minus_lora": paired("pissa", base_lr, r, "lora", base_lr, r, step)}
        for r in protocol["ranks"] for step in checkpoints]
    best = {}
    for method in METHODS:
        candidates = []
        for lr in protocol["lrs"]:
            values = [index[(s, method, lr, base_r, checkpoints[-1])]["validation_loss"] for s in seeds]
            candidates.append((statistics.mean(values), lr, values))
        score, lr, values = min(candidates)
        best[method] = {"lr": lr, "mean_final_validation_loss": score,
                        "per_seed_final_validation_loss": dict(zip(map(str, seeds), values))}
    selected = [{"step": step, "pissa_lr": best["pissa"]["lr"], "fc_full_lr": best["fc_full"]["lr"],
        **paired("pissa", best["pissa"]["lr"], base_r, "fc_full", best["fc_full"]["lr"], base_r, step)}
        for step in checkpoints]
    trends = []
    for step in checkpoints:
        for axis, series in (("lr", sorted((x["lr"], x["mean_signed_gap"]) for x in same_lr if x["step"]==step)),
                             ("rank", sorted((x["r"], x["pissa_minus_fc_full"]["mean_signed_gap"])
                                             for x in rank_rows if x["step"]==step))):
            for (before, gap_before), (after, gap_after) in zip(series, series[1:]):
                change = gap_after-gap_before
                trends.append({"axis": axis, "step": step, "from": before, "to": after,
                    "change_in_mean_signed_gap": change,
                    "direction": "increased" if change>0 else "decreased" if change<0 else "unchanged"})
    return {"same_lr_pissa_minus_fc_full": same_lr, "rank_comparisons": rank_rows,
        "best_lr_by_method": best, "selected_lr_pissa_minus_fc_full": selected,
        "signed_gap_trends": trends,
        "interpretation": "descriptive, validation-selected, exploratory; no statistical significance or causal claim"}


def run_scan(args, datasets=None):
    """执行训练集内部验证扫描，保存协议、逐单元结果、比较与诊断图，返回退出码。"""
    config = {**vars(args), "output": str(args.output.resolve()), "data_dir": str(DATA_DIR),
        "mode": "diagnostic_scan", "data_source": "official_MNIST_train_only" if datasets is None else "injected_test_fixture",
        "storage_dtype": "torch.float32", "main_experiment_changed": False}
    path = args.output / "metrics/mnist_scan.json"
    started = time.perf_counter()
    report = new_report("M0/mnist_scan", config, args.device)
    report["protocol"] = scan_protocol(args)
    report["results"] = {"runs": []}
    atomic_json(path, report)  # fixed comparison/selection rules precede every run
    try:
        train = datasets[0] if datasets is not None else None
        grid = scan_grid(args.scan_lrs, args.scan_ranks, args.r, args.lr)
        report["config"]["grid"] = grid
        for seed in args.seeds:
            bundle = seed_bundle(seed)
            loaders = get_scan_loaders(args.batch, seed=bundle["pretrain_data"],
                split_seed=args.split_seed, validation_fraction=args.validation_fraction, train_dataset=train)
            train = loaders["even"].dataset.dataset
            with seeded(bundle["base"], args.device):
                base = MLP(args.hidden).to(args.device)
            tick = time.perf_counter()
            pretraining = pretrain(base, loaders["odd"], args.epochs, args.device)
            run = {"seed": seed, "seed_bundle": bundle, "split": loaders["split"],
                "pretraining": pretraining, "pretrain_seconds": time.perf_counter()-tick, "experiments": []}
            report["results"]["runs"].append(run)
            probe = torch.stack([loaders["even"].dataset[i][0] for i in range(min(8, len(loaders["even"].dataset)))]).to(args.device)
            atomic_json(path, report)
            for cell in grid:
                tick = time.perf_counter()
                with seeded(bundle["adapter"], args.device):
                    model, info = prepare_method(base, cell["method"], cell["r"], probe)
                loader = make_loader(loaders["even"].dataset, args.batch, bundle["finetune_data"], True)
                result = finetune(model, loader, args.ft_steps, cell["lr"], args.device,
                    loaders["validation"], args.eval_every, args.tail_window,
                    evaluation_steps=args.scan_checkpoints, evaluation_role="even training validation; official test not loaded")
                result.update(info, seconds=time.perf_counter()-tick)
                run["experiments"].append({**cell, "result": result})
                print(f"scan seed={seed} {cell['method']} lr={cell['lr']:g} r={cell['r']}: validation loss={result['evaluations'][-1]['loss']:.5f}", flush=True)
                atomic_json(path, report)
                del model
            del base
        table = scan_table(report["results"]["runs"], args.scan_checkpoints)
        report["results"]["table"] = table
        report["results"]["comparisons"] = scan_comparisons(table, report["protocol"])
        csv_path = args.output / "metrics/mnist_scan.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        report["config"]["table_csv"] = str(csv_path.resolve())
        print("seed,method,lr,r,step,validation_loss,fc_step_displacement", flush=True)
        for row in table:
            print(f"{row['seed']},{row['method']},{row['lr']:g},{row['r']},{row['step']},{row['validation_loss']:.8f},{row['fc_step_displacement']:.8g}", flush=True)
        code = finish_report(report, path, "passed", 0, started)
        print(f"Diagnostic scan completed: {path.resolve()}\nFull table: {csv_path.resolve()}", flush=True)
        return code
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finish_report(report, path, "error", 1, started)
        print(f"MNIST scan error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main(argv=None, *, datasets=None):
    """解析参数并执行 M0 主实验或验证扫描，保存报告与图，返回退出码。

    Args:
        argv: 可选命令行参数列表；None 时读取进程参数。
        datasets: 可选测试数据对；注入假数据时明确标记为合成 smoke 证据。

    Returns:
        成功为 0，执行失败为 1；非法参数由解析器以 64 退出。
    """
    parser = ArgumentParser64(description="Paired odd-to-even MNIST mechanism experiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ft-steps", type=int, default=300)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--tail-window", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scan", action="store_true", help="validation-only one-factor diagnostic, separate output required")
    parser.add_argument("--scan-lrs", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--scan-ranks", nargs="+", type=int, default=[8, 32, 128, 256])
    parser.add_argument("--scan-checkpoints", nargs=2, type=int, default=[50, 300])
    parser.add_argument("--split-seed", type=int, default=20261101)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    args = parser.parse_args(argv)
    if min(args.epochs, args.ft_steps, args.r, args.hidden, args.batch, args.eval_every, args.tail_window, args.threads)<=0:
        parser.error("counts must be positive")
    if args.r>args.hidden or not math.isfinite(args.lr) or args.lr<=0 or len(set(args.seeds))!=len(args.seeds):
        parser.error("require r <= hidden, finite positive lr and distinct seeds")
    torch.set_num_threads(args.threads)
    if args.scan:
        if args.output.resolve() == DEFAULT_OUTPUT.resolve():
            parser.error("--scan requires a separate --output; formal results must not be overwritten")
        if not 0 < args.validation_fraction < 1:
            parser.error("validation fraction must lie in (0,1)")
        if any(not math.isfinite(lr) or lr<=0 for lr in args.scan_lrs) or args.lr not in args.scan_lrs:
            parser.error("scan learning rates must be positive/finite and include base --lr")
        if any(r<=0 or r>args.hidden for r in args.scan_ranks) or args.r not in args.scan_ranks:
            parser.error("scan ranks must be within hidden size and include base --r")
        if len(set(args.scan_lrs))!=len(args.scan_lrs) or len(set(args.scan_ranks))!=len(args.scan_ranks):
            parser.error("scan values must be distinct")
        if not 0 < args.scan_checkpoints[0] < args.scan_checkpoints[1] == args.ft_steps:
            parser.error("scan checkpoints must increase and end at --ft-steps")
        return run_scan(args, datasets)
    config = {**vars(args), "output": str(args.output.resolve()), "data_dir": str(DATA_DIR),
        "data_source": "official_MNIST" if datasets is None else "injected_test_fixture",
        "methods": list(METHODS), "lora_A_std": 1/args.r, "storage_dtype": "torch.float32",
        "test_usage": "predeclared evaluation; no hyperparameter/checkpoint/seed selection"}
    started = time.perf_counter()
    path = args.output / "metrics/mnist.json"
    report = new_report("M0/mnist_toy", config, args.device)
    report["results"] = {"runs": []}
    atomic_json(path, report)
    try:
        train, test = datasets if datasets is not None else (None, None)
        for seed in args.seeds:
            bundle = seed_bundle(seed)
            tick = time.perf_counter()
            loaders = get_loaders(args.batch, seed=bundle["pretrain_data"], train_dataset=train, test_dataset=test)
            # Reuse actual datasets, but never reuse the progressing training generator.
            train = loaders["odd"].dataset.dataset
            test = loaders["test"].dataset.dataset
            data_seconds = time.perf_counter()-tick
            with seeded(bundle["base"], args.device):
                base = MLP(args.hidden).to(args.device)
            tick = time.perf_counter()
            pretraining = pretrain(base, loaders["odd"], args.epochs, args.device)
            pretrain_seconds = time.perf_counter()-tick
            probe = torch.stack([loaders["even"].dataset[i][0] for i in range(min(8, len(loaders["even"].dataset)))]).to(args.device)
            run = {"seed": seed, "seed_bundle": bundle, "pretraining": pretraining,
                "data_seconds": data_seconds, "pretrain_seconds": pretrain_seconds, "methods": {}}
            report["results"]["runs"].append(run)
            atomic_json(path, report)
            for method in METHODS:
                tick = time.perf_counter()
                with seeded(bundle["adapter"], args.device):
                    model, info = prepare_method(base, method, args.r, probe)
                loader = make_loader(loaders["even"].dataset, args.batch, bundle["finetune_data"], True)
                result = finetune(model, loader, args.ft_steps, args.lr, args.device,
                                  loaders["test"], args.eval_every, args.tail_window)
                result.update(info, seconds=time.perf_counter()-tick)
                run["methods"][method] = result
                print(f"seed={seed} {method}: final test loss={result['evaluations'][-1]['loss']:.5f}", flush=True)
                atomic_json(path, report)
            del model, base
        report["results"]["summary"] = paired_summary(report["results"]["runs"])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = args.output / "figures/mnist_toy_convergence.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(args.seeds), figsize=(6*len(args.seeds), 4), squeeze=False)
        for ax, run in zip(axes[0], report["results"]["runs"]):
            for method in METHODS:
                ax.plot(range(1, args.ft_steps+1), run["methods"][method]["loss_curve"], label=LABELS[method])
            title = "MNIST mechanism" if datasets is None else "SYNTHETIC SMOKE — not MNIST evidence"
            ax.set(title=f"{title}, seed={run['seed']}", xlabel="Optimizer step", ylabel="Online training loss")
            ax.legend()
        fig.tight_layout()
        fig.savefig(figure, dpi=150)
        plt.close(fig)
        report["config"]["figure"] = str(figure.resolve())
        return finish_report(report, path, "passed", 0, started)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finish_report(report, path, "error", 1, started)
        print(f"MNIST error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
