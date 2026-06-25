"""[M0] 复现论文 Fig.2a 的玩具实验：PiSSA 收敛更快。

流程：
  1. 用一个两层 MLP 在 MNIST 的「奇数类」上预训练；
  2. 冻结后分别用 Full-FT / LoRA / PiSSA 在「偶数类」上微调；
  3. 对比三者的微调 loss 曲线，PiSSA 应更快、更低（对应论文 Fig.2a）。

依赖 torchvision 下载 MNIST。用法：
    python scripts/m0/mnist_toy.py --epochs 3 --ft-steps 200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pissa import PiSSALinear  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "m0" / "exp1" / "figures"
METRICS_DIR = REPO_ROOT / "results" / "m0" / "exp1" / "metrics"
DATA_DIR = REPO_ROOT / "data" / "mnist"


class MLP(nn.Module):
    """两层 MLP，中间一层是被适配的目标线性层 ``fc``。"""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.enc = nn.Linear(28 * 28, hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.enc(x))
        x = F.relu(self.fc(x))
        return self.head(x)


class LoRALinear(nn.Module):
    """对照用的标准 LoRA（A 高斯、B 零），冻结原权重。"""

    def __init__(self, linear: nn.Linear, r: int, alpha: float | None = None) -> None:
        super().__init__()
        self.r = r
        self.scaling = (alpha or r) / r
        self.register_buffer("weight", linear.weight.data.clone())
        self.register_buffer(
            "bias", linear.bias.data.clone() if linear.bias is not None else None
        )
        self.lora_A = nn.Parameter(torch.randn(r, linear.in_features) * (1.0 / r**0.5))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        return base + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)


def get_loaders(batch: int):
    from torchvision import datasets, transforms

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=tfm)
    odd_idx = [i for i, (_, y) in enumerate(train) if y % 2 == 1]
    even_idx = [i for i, (_, y) in enumerate(train) if y % 2 == 0]
    odd = torch.utils.data.Subset(train, odd_idx)
    even = torch.utils.data.Subset(train, even_idx)
    return (
        torch.utils.data.DataLoader(odd, batch_size=batch, shuffle=True),
        torch.utils.data.DataLoader(even, batch_size=batch, shuffle=True),
    )


def pretrain(model: MLP, loader, epochs: int, device: str) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        print(f"  预训练 epoch {ep + 1}/{epochs} 末 loss={loss.item():.4f}")


def finetune(model: MLP, loader, method: str, steps: int, r: int, lr: float, device: str):
    """返回每步 loss 列表。method ∈ {full, lora, pissa}。"""
    for p in model.parameters():
        p.requires_grad_(False)

    if method == "full":
        model.fc.weight.requires_grad_(True)
        model.fc.bias.requires_grad_(True)
        params = [model.fc.weight, model.fc.bias]
    elif method == "lora":
        model.fc = LoRALinear(model.fc, r=r).to(device)
        params = [model.fc.lora_A, model.fc.lora_B]
    elif method == "pissa":
        model.fc = PiSSALinear.from_linear(model.fc, r=r, alpha=r).to(device)
        params = [model.fc.lora_A, model.fc.lora_B]
    else:
        raise ValueError(method)

    opt = torch.optim.Adam(params, lr=lr)
    losses: list[float] = []
    model.train()
    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="[M0] MNIST 玩具实验（Fig.2a）")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ft-steps", type=int, default=200)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    import copy

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{device}")

    odd_loader, even_loader = get_loaders(args.batch)

    print("在奇数类上预训练基础模型 ...")
    base = MLP().to(device)
    pretrain(base, odd_loader, args.epochs, device)

    curves = {}
    for method in ("full", "lora", "pissa"):
        print(f"在偶数类上微调：{method} ...")
        model = copy.deepcopy(base).to(device)
        curves[method] = finetune(
            model, even_loader, method, args.ft_steps, args.r, args.lr, device
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for method, losses in curves.items():
        plt.plot(losses, label=method.upper())
    plt.xlabel("fine-tuning step")
    plt.ylabel("loss")
    plt.title("MNIST toy: convergence (odd→even), reproduce Fig.2a")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "mnist_toy_convergence.png"
    plt.savefig(out_path, dpi=150)
    print(f"已保存：{out_path}")
    for method, losses in curves.items():
        print(f"  {method:5s} 末步 loss = {losses[-1]:.4f}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "mnist.json"
    payload = {
        "experiment": "M0/mnist_toy (Fig.2a)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "epochs": args.epochs,
            "ft_steps": args.ft_steps,
            "r": args.r,
            "lr": args.lr,
            "batch": args.batch,
            "device": device,
            "figure": str(out_path),
        },
        "results": {
            "loss_curves": curves,
            "final_loss": {m: losses[-1] for m, losses in curves.items()},
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存 metrics：{metrics_path}")


if __name__ == "__main__":
    main()
