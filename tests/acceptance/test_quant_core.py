"""量化核心验收（新增文件）：QPiSSA 交替初始化、Linear4bit 基座、QLoRA、合并与恢复。

需要 CUDA 与 bitsandbytes（真实 NF4 内核，不使用假后端）。无 CUDA 时跳过并明确标注。
"""
import gc
import weakref

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能运行真实 bnb NF4 内核")

CUDA = "cuda"


def _weight(out=192, inn=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(out, out, generator=g))
    v, _ = torch.linalg.qr(torch.randn(inn, inn, generator=g))
    s = torch.linspace(5.0, 0.05, min(out, inn)) ** 2
    k = min(out, inn)
    return (u[:, :k] * s) @ v[:k, :]


def _rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def _toy(w):
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(w.shape[1], w.shape[0], bias=False)
            self.q_proj.weight.data.copy_(w)

    return Toy().to(CUDA)


def test_alternating_init_reconstruction_and_round_naming():
    from pissa.layer_nf4 import qpissa_alternating_init
    from pissa.svd_init import pissa_init

    w = _weight().to(CUDA)
    one = qpissa_alternating_init(w, r=16, rounds=1)
    five = qpissa_alternating_init(w, r=16, rounds=5)
    assert len(one.round_errors) == 1 and len(five.round_errors) == 5
    # rounds=1 等于普通 PiSSA 主成分
    ref = pissa_init(w, r=16)
    assert _rel(one.lora_B @ one.lora_A, ref.lora_B @ ref.lora_A) < 1e-5
    # 最终重构 W ≈ Q_T + BA，误差等于末轮记录的量化误差
    for init in (one, five):
        err = _rel(init.quantized_residual + init.lora_B @ init.lora_A, w)
        assert err == pytest.approx(init.round_errors[-1], rel=1e-3)
    assert one.lora_A.dtype == torch.float32 and five.quantized_residual.dtype == torch.float32
    with pytest.raises(ValueError):
        qpissa_alternating_init(w, r=16, rounds=0)


def test_qpissa_layer_storage_forward_and_report():
    from adapters.quant import dequantize_linear4bit
    from pissa.layer_nf4 import NFPiSSALinear

    w = _weight().to(CUDA)
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False).to(CUDA)
    lin.weight.data.copy_(w)
    layer = NFPiSSALinear.from_linear(lin, r=16, alpha=16, rounds=2)
    # 没有持久浮点残差 buffer；浮点张量只有 A/B
    assert list(layer.buffers()) == []
    floats = [(n, p.dtype) for n, p in layer.named_parameters() if p.is_floating_point()]
    assert sorted(n for n, _ in floats) == ["lora_A", "lora_B"] and all(d == torch.float32 for _, d in floats)
    assert layer.base.weight.dtype == torch.uint8 and not layer.base.weight.requires_grad
    rep = layer.init_report
    assert rep["rounds"] == 2 and len(rep["round_errors"]) == 2
    # 训练基座由 BF16 + double quant 重新量化，一般不等于 Q_T：偏差为正但很小
    assert 0 < rep["requant_rel_dev_vs_Q_T"] < 0.2
    assert rep["recon_rel_err_vs_W"] < 0.3
    # 前向对应 dequant(base) + BA
    x = torch.randn(4, w.shape[1], device=CUDA)
    layer.lora_B.data.normal_(0, 0.05)  # 非零低秩贡献
    y = layer(x)
    expect = x @ layer.merged_weight().T
    assert _rel(y, expect) < 3e-2
    assert (layer.merged_weight() - dequantize_linear4bit(layer.base).float()).abs().max() > 0


def test_qpissa_gradients_and_frozen_base():
    from pissa.layer_nf4 import NFPiSSALinear

    w = _weight().to(CUDA)
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False).to(CUDA)
    lin.weight.data.copy_(w)
    layer = NFPiSSALinear.from_linear(lin, r=8, alpha=8)
    x = torch.randn(3, w.shape[1], device=CUDA, requires_grad=True)
    layer(x).square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0  # base 没有切断输入梯度
    assert layer.lora_A.grad.abs().sum() > 0 and layer.lora_B.grad.abs().sum() > 0
    assert layer.base.weight.grad is None


def test_qpissa_rejects_bad_inputs():
    from pissa.layer_nf4 import NFPiSSALinear

    w = _weight().to(CUDA)
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False).to(CUDA)
    with pytest.raises(ValueError):
        NFPiSSALinear.from_linear(lin, r=8, alpha=16)  # alpha != r
    with pytest.raises(ValueError):
        NFPiSSALinear.from_linear(lin, r=True)
    with pytest.raises(ValueError):
        NFPiSSALinear.from_linear(lin, r=10_000)  # 超过 min(in,out)


def test_inject_merge_qpissa_and_lifetime():
    from pissa.inject_nf4 import inject_qpissa, merge_qpissa
    from pissa.layer_nf4 import NFPiSSALinear

    w = _weight().to(CUDA)
    model = _toy(w)
    ref = weakref.ref(model.q_proj)
    reports = {}
    names = inject_qpissa(model, 8, 8, ("q_proj",), rounds=2, reports=reports)
    assert names == ["q_proj"] and isinstance(model.q_proj, NFPiSSALinear) and "q_proj" in reports
    gc.collect()
    assert ref() is None, "旧浮点层仍被引用"
    model.q_proj.lora_B.data.normal_(0, 0.05)
    before = model.q_proj.merged_weight().clone()
    merge_qpissa(model)
    assert type(model.q_proj) is nn.Linear and model.q_proj.weight.dtype == torch.float32
    assert torch.equal(model.q_proj.weight.data, before)


def test_inject_qpissa_rejects_quantized_and_shared_targets():
    from adapters.quant import build_linear4bit
    from pissa.inject_nf4 import inject_qpissa

    w = _weight().to(CUDA)
    model = _toy(w)
    model.q_proj = build_linear4bit(w, None, torch.bfloat16, False, CUDA)
    with pytest.raises(TypeError):
        inject_qpissa(model, 8, 8, ("q_proj",))
    model2 = _toy(w)
    model2.k_proj = model2.q_proj  # 同一对象两个名字
    with pytest.raises(ValueError):
        inject_qpissa(model2, 8, 8, ("q_proj", "k_proj"))


def test_qlora_layer_inject_unwrap_and_restore():
    from adapters.quant import build_linear4bit
    from lora.inject_nf4 import inject_lora_4bit, unwrap_qlora
    from lora.layer_nf4 import QLoRALinear
    from scripts.m1.eval_gsm8k import validate_qlora_restore

    w = _weight().to(CUDA)
    model = _toy(w)
    base = build_linear4bit(w, None, torch.bfloat16, True, CUDA)
    model.q_proj = base
    x = torch.randn(4, w.shape[1], device=CUDA)
    with torch.no_grad():
        y0 = base(x)
    names = inject_lora_4bit(model, 8, 32, ("q_proj",), dtype=torch.float32)  # alpha != r 对 LoRA 合法
    layer = model.q_proj
    assert names == ["q_proj"] and isinstance(layer, QLoRALinear) and layer.scaling == 4.0
    assert layer.lora_A.dtype == torch.float32 and layer.base is base
    with torch.no_grad():
        assert torch.equal(layer(x), y0)  # B=0 时等于实际量化基座
        layer.lora_B.normal_(0, 0.05)
        y1 = layer(x)
    assert not torch.allclose(y1, y0)
    sd = {}
    unwrap_qlora(model, sd)
    assert model.q_proj is base and set(sd) == {"q_proj.lora_A", "q_proj.lora_B"}
    assert sd["q_proj.lora_A"].device.type == "cpu" and sd["q_proj.lora_A"].dtype == torch.float32
    # 恢复：重新注入并校验、加载
    model2 = _toy(w)
    model2.q_proj = base
    replaced = inject_lora_4bit(model2, 8, 32, ("q_proj",), dtype=torch.float32)
    bad = {k: v for k, v in sd.items() if k != "q_proj.lora_B"}
    with pytest.raises(RuntimeError):
        validate_qlora_restore(model2, replaced, bad)
    with pytest.raises(RuntimeError):
        validate_qlora_restore(model2, [], sd)
    validate_qlora_restore(model2, replaced, sd)
    with torch.no_grad():
        assert torch.allclose(model2.q_proj(x), y1)


def test_qlora_rejects_non_4bit_and_bad_rank():
    from adapters.quant import build_linear4bit
    from lora.inject_nf4 import inject_lora_4bit
    from lora.layer_nf4 import QLoRALinear

    w = _weight().to(CUDA)
    with pytest.raises(TypeError):
        inject_lora_4bit(_toy(w), 8, 8, ("q_proj",))
    base = build_linear4bit(w, None, torch.bfloat16, False, CUDA)
    for bad_r in (0, -1, True):
        with pytest.raises(ValueError):
            QLoRALinear(base, r=bad_r)
    with pytest.raises(ValueError):
        QLoRALinear(base, r=8, alpha=0)
    with pytest.raises(TypeError):
        QLoRALinear(nn.Linear(4, 4), r=2)


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(CUDA).float()


def _cfg(rounds=2):
    return {
        "adapter": {"r": 8, "alpha": 8, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"],
                    "niter": None, "init_rounds": rounds},
        "quantization": {"bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True,
                         "init_double_quant": False},
        "methods": {"qpissa": {"adapter_dtype": "float32"}, "qlora": {"adapter_dtype": "float32"},
                    "pissa": {}, "lora": {}},
    }


@pytest.mark.parametrize("method", ["pissa", "qpissa"])
def test_apply_adapter_end_to_end_with_checkpointing(method):
    """整模型路径：注入、冻结范围、梯度检查点下适配器仍有梯度、QPiSSA 合并后形态正确。"""
    from scripts.model_ops import apply_adapter, merge_adapter

    model = _tiny_llama()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    stats = apply_adapter(model, method, _cfg())
    assert stats["replaced_modules"] == 14
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(("lora_A" in n or "lora_B" in n) for n in trainable)
    if method == "qpissa":
        agg = stats["qpissa_init"]
        assert agg["num_layers"] == 14 and agg["requant_rel_dev_vs_Q_T"]["max"] > 0
    model.train()
    ids = torch.randint(0, 128, (2, 16), device=CUDA)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    merge_adapter(model, method)
    assert not any(type(m).__name__ in ("NFPiSSALinear", "PiSSALinear", "Linear4bit") for m in model.modules())
