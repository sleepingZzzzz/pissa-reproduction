"""FP32 主权重 AdamW（CPUMasterAdamW）验收（新增文件）。

CPU 上即可验证数值语义；GPU 上的锁页内存/流水线路径需在有卡机器上另测（此文件在有卡时也会跑 CUDA 版本）。
"""
import pytest
import torch

from scripts.model_ops import CPUMasterAdamW, build_optimizer


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make(device, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = [(8, 5), (7,), (3, 4)][:n]
    base = [torch.randn(s, generator=g) for s in shapes]
    bf = [b.to(torch.bfloat16).to(device).requires_grad_(True) for b in base]
    return bf


@pytest.mark.parametrize("device", _devices())
def test_matches_torch_adamw_on_fp32_master(device):
    """主权重轨迹应与 torch.optim.AdamW（FP32 参数副本）一致，BF16 参数 = 主权重舍入。"""
    bf = _make(device)
    ref = [p.detach().float().clone().requires_grad_(True) for p in bf]
    opt = CPUMasterAdamW(bf, lr=1e-3, weight_decay=0.1)
    ref_opt = torch.optim.AdamW(ref, lr=1e-3, weight_decay=0.1)
    g = torch.Generator().manual_seed(1)
    for step in range(6):
        for p, r in zip(bf, ref):
            grad = torch.randn(p.shape, generator=g)
            p.grad = grad.to(torch.bfloat16).to(device)
            r.grad = p.grad.float().clone()
        opt.param_groups[0]["lr"] = 1e-3 * (1 - step / 10)
        ref_opt.param_groups[0]["lr"] = 1e-3 * (1 - step / 10)
        opt.step()
        ref_opt.step()
        opt.zero_grad()
        for i, (p, r) in enumerate(zip(bf, ref)):
            assert torch.allclose(opt.master[i].to(device), r.detach(), rtol=1e-5, atol=1e-7)
            assert torch.equal(p.detach(), r.detach().to(torch.bfloat16))
            assert p.grad is None


@pytest.mark.parametrize("device", _devices())
def test_small_updates_accumulate_unlike_pure_bf16(device):
    """lr 远小于 BF16 间隔时：纯 BF16 AdamW 权重不动，FP32 主权重累积后 BF16 参数会移动。"""
    w0 = torch.ones(64, dtype=torch.bfloat16, device=device)
    pure = w0.clone().requires_grad_(True)
    mast = w0.clone().requires_grad_(True)
    pure_opt = torch.optim.AdamW([pure], lr=1e-4, weight_decay=0.0)
    mast_opt = CPUMasterAdamW([mast], lr=1e-4, weight_decay=0.0)
    for _ in range(100):
        for p in (pure, mast):
            p.grad = torch.ones_like(p)
        pure_opt.step()
        mast_opt.step()
    assert torch.equal(pure.detach(), w0)  # 每步 1e-4 < 间隔一半(≈0.0039)：全部丢失
    assert (mast.detach().float() != 1.0).all()  # 累积约 1e-2，越过舍入阈值
    assert mast_opt.master[0].mean().item() == pytest.approx(1.0 - 1e-2, abs=1e-4)


def test_none_gradient_params_are_skipped():
    a, b = _make("cpu", n=2)
    opt = CPUMasterAdamW([a, b], lr=1e-2, weight_decay=0.0)
    b0 = b.detach().clone()
    a.grad = torch.ones_like(a)
    opt.step()
    assert torch.equal(b.detach(), b0)
    assert not torch.equal(a.detach(), a.detach() * 0 + _make("cpu", n=2)[0].detach())


def test_rejects_non_bf16_and_bad_modes():
    with pytest.raises(RuntimeError):
        CPUMasterAdamW([torch.zeros(3, requires_grad=True)], lr=1e-3, weight_decay=0.0)
    with pytest.raises(RuntimeError):
        CPUMasterAdamW([], lr=1e-3, weight_decay=0.0)
    p = _make("cpu", n=1)
    with pytest.raises(ValueError):
        build_optimizer(p, 1e-3, 0.0, "adamw", master_weights="bogus")
    with pytest.raises(ValueError):
        build_optimizer(p, 1e-3, 0.0, "adamw_8bit", master_weights="cpu_fp32")
    opt, name = build_optimizer(p, 1e-3, 0.0, "adamw", master_weights="cpu_fp32")
    assert name == "adamw_cpu_fp32_master" and isinstance(opt, CPUMasterAdamW)
    opt, name = build_optimizer(_make("cpu", n=1), 1e-3, 0.0, "adamw")
    assert name == "adamw" and isinstance(opt, torch.optim.AdamW)


def test_available_host_bytes_counts_reclaimable_page_cache(tmp_path):
    from scripts.model_ops import _available_host_bytes

    gib = 2**30
    (tmp_path / "memory.max").write_text(str(110 * gib))
    (tmp_path / "memory.current").write_text(str(35 * gib))
    (tmp_path / "memory.stat").write_text(f"anon {4 * gib}\nfile {31 * gib}\ninactive_file {16 * gib}\nactive_file {15 * gib}\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {800 * gib // 1024} kB\n")
    # 110 - (35 - 31) = 106 GiB，而不是 110 - 35 = 75 GiB
    assert _available_host_bytes(str(tmp_path), str(meminfo)) == 106 * gib
    (tmp_path / "memory.stat").unlink()  # 没有 memory.stat 时退回保守估计
    assert _available_host_bytes(str(tmp_path), str(meminfo)) == 75 * gib
    (tmp_path / "memory.max").write_text("max")
    assert _available_host_bytes(str(tmp_path), str(meminfo)) == 800 * gib
