"""M0 odd-to-even MNIST mechanism experiment, not a Fig.2(a) trajectory replica.

Three Linear layers, two hidden ReLUs. Each seed retrains one shared base;
four paired methods see identical batches. Test data never selects settings.
"""
from __future__ import annotations

import copy
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
    """784 -> hidden -> hidden -> 10; head returns logits, not probabilities."""
    def __init__(self, hidden=256):
        super().__init__()
        self.enc = nn.Linear(784, hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.head(F.relu(self.fc(F.relu(self.enc(x.flatten(1))))))


class LoRALinear(nn.Module):
    """Gaussian std=1/r A, zero B; frozen base and bias, no dropout."""
    def __init__(self, linear, r, alpha=None):
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
        return F.linear(x, self.weight, self.bias)+self.scaling*F.linear(F.linear(x, self.lora_A), self.lora_B)


def seed_bundle(seed):
    return {"base": seed, "pretrain_data": seed+100_003,
            "adapter": seed+200_003, "finetune_data": seed+300_007}


def make_loader(dataset, batch, seed, shuffle):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0, drop_last=False)


def batch_stream(loader):
    if not len(loader):
        raise ValueError("empty training loader")
    while True:
        yield from loader


def get_loaders(batch, *, seed=0, train_dataset=None, test_dataset=None):
    if train_dataset is None or test_dataset is None:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307,), (.3081,))])
        # Data cache is independent of --output. No test fixture reaches here.
        train_dataset = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)
    def subset(dataset, parity):
        labels = torch.as_tensor(dataset.targets)
        indices = (labels % 2 == parity).nonzero(as_tuple=True)[0].tolist()
        if not indices:
            raise ValueError("empty odd/even dataset")
        return Subset(dataset, indices)
    return {"odd": make_loader(subset(train_dataset, 1), batch, seed, True),
            "even": make_loader(subset(train_dataset, 0), batch, seed, True),
            "test": make_loader(subset(test_dataset, 0), batch, seed, False)}


def validate_initialization(base, model, method, r, probe):
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
    return {**{n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad},
            **{n: b.detach().clone() for n, b in model.named_buffers()}}


def validate_frozen(model, before):
    current = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    if any(n not in current or not torch.equal(value, current[n]) for n, value in before.items()):
        raise ValueError("frozen parameter/buffer changed")


@torch.no_grad()
def evaluate(model, loader, device):
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


def finetune(model, loader, steps, lr, device, eval_loader, eval_every, tail_window):
    before = frozen_state(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    stream, losses, evaluations, sizes = batch_stream(loader), [], [], []
    tick = time.perf_counter()
    evaluations.append({"step": 0, **evaluate(model, eval_loader, device)})
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
        losses.append(float(loss.detach()))
        sizes.append(len(y))
        if step % eval_every == 0 or step == steps:
            tick = time.perf_counter()
            evaluations.append({"step": step, **evaluate(model, eval_loader, device)})
            eval_seconds += time.perf_counter()-tick
    validate_frozen(model, before)
    n = min(tail_window, steps)
    return {"loss_curve": losses, "last_online_loss": losses[-1],
        "tail_online_mean": sum(a*b for a, b in zip(losses[-n:], sizes[-n:]))/sum(sizes[-n:]),
        "tail_window": n, "evaluations": evaluations, "frozen_unchanged": True,
        "samples_seen": sum(sizes), "evaluation_seconds": eval_seconds,
        "loop_seconds": time.perf_counter()-started,
        "loss_semantics": "online pre-update batch loss; tail mixes parameter states"}


def check_lora_peft(device="cpu"):
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


def main(argv=None, *, datasets=None):
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
    args = parser.parse_args(argv)
    if min(args.epochs, args.ft_steps, args.r, args.hidden, args.batch, args.eval_every, args.tail_window, args.threads)<=0:
        parser.error("counts must be positive")
    if args.r>args.hidden or not math.isfinite(args.lr) or args.lr<=0 or len(set(args.seeds))!=len(args.seeds):
        parser.error("require r <= hidden, finite positive lr and distinct seeds")
    torch.set_num_threads(args.threads)
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
