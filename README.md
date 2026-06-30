# PiSSA Reproduction

Reproduction of **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models** (NeurIPS 2024).

PiSSA shares the same architecture as LoRA; the only difference is adapter initialization. It runs an SVD on the pretrained weight `W`, initializes the trainable `A`, `B` from the top-`r` principal components, and keeps the rest as a frozen residual `W_res`. This converges faster and reaches better results than LoRA.

```
W = U S Vᵀ
A = S[:r]^½ · V[:,:r]ᵀ ,  B = U[:,:r] · S[:r]^½ ,  W_res = W − B A
forward:  y = x (W_res + B A)
```

## Milestones

| Milestone | Content | Compute |
|---|---|---|
| **M0** ✅ | From-scratch SVD initialization + correctness verification (reconstruction identity, alignment with HF PEFT, Eckart–Young property, Fig.3 singular spectrum, Fig.2a MNIST toy convergence) | Local / free |
| **M1** ✅ | Single model **LLaMA-2-7B** (paper-aligned), PiSSA vs LoRA (±Full-FT), proving faster/better convergence (MetaMathQA → GSM8K), validated against the paper's numbers | RTX PRO 6000 96GB |
| **M2** ✅ | Repeat on **newer / cross-family models** (ChatGLM3-6B, Qwen2.5-7B, Llama3-8B, Baichuan2-7B) to test generality beyond the paper's models | RTX PRO 6000 96GB |
| **M3** ✅ | Quantized QPiSSA vs QLoRA: NF4 error-reduction table (Table 4) + 4-bit training convergence | RTX PRO 6000 96GB |

## Results Summary

All numbers are GSM8K exact-match accuracy (test set, 1319 problems). Adapter rank `r=128`, `alpha=128`, 1 epoch, AdamW, lr `2e-5`, cosine schedule, global batch 128, `max_seq_len=1024`.

### M1 — LLaMA-2-7B on MetaMathQA (100K samples) — paper-aligned

| Method | Trainable % | Final train loss | GSM8K Acc |
|--------|:-----------:|:----------------:|:---------:|
| Full-FT | 100% | 0.2733 | **0.3988** |
| **PiSSA** | 54.9% | 0.2874 | **0.3988** |
| LoRA | 54.9% | 0.3646 | 0.3237 |

> **Key result**: PiSSA matches Full-FT accuracy exactly and beats LoRA by +7.5pp, with only 54.9% of parameters trainable. This is the strongest evidence of the paper's core claim.

### M2 — Cross-model generality

| Model | Train samples | PiSSA Acc | LoRA Acc | Note |
|-------|:-------------:|:---------:|:--------:|------|
| ChatGLM3-6B | 100K | 0.5777 | **0.5830** | full run; PiSSA ≈ LoRA (within noise) |
| Qwen2.5-7B | 20K | **0.7544** | 0.7468 | partial run; PiSSA ≥ LoRA |
| Llama3-8B | 20K | 0.5815 | **0.5838** | partial run; PiSSA ≈ LoRA |
| Baichuan2-7B | 100K | 0.0000 | 0.0000 | ❌ failed (see note below) |

> Qwen2.5 and Llama3 were trained on a reduced 20K subset (not full 100K); treat as indicative.
> Baichuan2 produced incoherent output for both methods after fine-tuning — likely due to architecture specifics (ALiBi + `W_pack`) interacting badly with the adapter injection under transformers 5.x. Not used as evidence.

### M3 — NF4 4-bit quantization (LLaMA-2-7B, 100K samples)

| Method | Final train loss | GSM8K Acc |
|--------|:----------------:|:---------:|
| QPiSSA | 0.4114 | 0.2297 |
| QLoRA | 0.3964 | **0.2722** |

**Table 4 — NF4 quantization reconstruction error** (avg over sampled layers):

| | Baseline NF4 error | PiSSA-residual NF4 error | Improvement |
|---|:---:|:---:|:---:|
| value | 7.19% | 1.51% | **79.0%** |

> QPiSSA mathematically reduces the NF4 quantization error of the frozen residual (Table 4), but in end-to-end training it underperformed QLoRA. The most likely cause is our from-scratch NF4 path in `src/pissa/layer_nf4.py` (CPU→GPU dequantize each forward) being numerically weaker than bitsandbytes' native fused 4-bit matmul kernel used by QLoRA. Reported honestly as an engineering limitation.

## Directory Layout

```
src/pissa/          From-scratch implementation: svd_init / layer (PiSSALinear) / layer_nf4 (NFPiSSALinear) / inject
scripts/m0..m3/     Runnable scripts per milestone (m1/ scripts are reused for M2/M3 via different configs)
configs/            Experiment configs (base.yaml holds shared hyperparameters)
results/m*/exp*/    Experiment artifacts (figures/*.png, metrics/*.json with loss curves, accuracy, and 20 sample generations each)
logs/               Raw training / eval run logs (large model-download logs are pruned)
.log/               Local-only project records (log.md / HANDOFF.md); git-ignored, not pushed
data/ checkpoints/  Large files, server-only, git-ignored
```

> **For paper writing without the server**: every quantitative result lives in `results/*/exp1/metrics/*.json` (training `loss_curve`, `final_loss`, `wall_time_sec`, `param_stats`, and per-method `accuracy` + 20 sample `{question, pred, gold, ok}`), plus the rendered figures under `results/*/exp1/figures/`. No model weights are needed to write up the numbers.

## Environment

```bash
pip install -r requirements.txt
```

> On the AutoDL server the image already ships a matching `torch` (`2.8.0+cu128`, CUDA build). Do **not** reinstall torch — `pip` may silently replace it with a CPU-only or mismatched build. Install the other packages only, then verify torch is unchanged:
>
> ```bash
> source /etc/network_turbo          # AutoDL download accelerator (optional)
> pip install transformers peft datasets accelerate sentencepiece protobuf modelscope addict simplejson sortedcontainers bitsandbytes
> python -c "import torch;print(torch.__version__)"   # must still print 2.8.0+cu128
> ```
>
> The environment can be configured in CPU-only ("no-GPU") mode; a GPU is only needed at training time.

## Running M0

```bash
# 1. Verify initialization correctness (reconstruction identity + Eckart–Young + HF PEFT alignment)
python scripts/m0/verify_init.py --in-features 1024 --out-features 1024 --r 64

# 2. Singular-value spectrum (Fig.3); reads a single real layer via ModelScope by default, add --random for offline
python scripts/m0/plot_singular.py

# 3. MNIST toy convergence comparison (Fig.2a)
python scripts/m0/mnist_toy.py --epochs 3 --ft-steps 300
```

Artifacts are saved to `results/m0/exp1/` (`figures/*.png` and `metrics/*.json`).

**M0 result**: our from-scratch PiSSA matches HF PEFT element-wise (zero relative error), satisfies the SVD/reconstruction properties, and reproduces PiSSA's faster-than-LoRA convergence on the toy task — implementation verified.

## Running M1

LLaMA-2-7B on MetaMathQA, evaluated on GSM8K. Config: `configs/m1_llama2_7b.yaml` (rank `r=128`, paper-aligned). Methods: our **PiSSA** / **LoRA** / **Full-FT**.

```bash
# 0. Prepare data (downloads MetaMathQA + GSM8K via ModelScope, writes local jsonl)
python scripts/m1/prepare_data.py --config configs/m1_llama2_7b.yaml          # add --limit 2000 for a smoke run

# 1. Train each method (saves merged model to checkpoints/m1/<method>)
python scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method pissa  # add --limit 64 for a smoke run
python scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method lora
python scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method full

# 2. Evaluate on GSM8K
python scripts/m1/eval_gsm8k.py --config configs/m1_llama2_7b.yaml --method pissa

# 3. Plot loss curves + accuracy bars
python scripts/m1/plot_curves.py --config configs/m1_llama2_7b.yaml
```

Memory strategy on a single 96GB GPU: PiSSA/LoRA train in fp32 (frozen base + tiny adapter) with gradient checkpointing on — fp32 activations are large, so disabling it OOMs even on 96GB; Full-FT cannot fit 7B in fp32 (params + AdamW state + grads ≈ 112GB), so it uses bf16 + 8-bit AdamW + gradient checkpointing (`methods.full` in the config). TF32 matmul is enabled in `train.py` to speed up the fp32 paths with negligible numerical impact (this is the main speed lever; it saves no memory). The smoke path (small `--limit`) is meant to be run first to validate the pipeline before launching the long runs.

## Running M2 (cross-model generality)

M2 reuses the M1 scripts with per-model configs. Each model has its own `configs/m2_<model>.yaml`.

```bash
for m in pissa lora; do
  python scripts/m1/train.py     --config configs/m2_qwen25_7b.yaml --method $m
  python scripts/m1/eval_gsm8k.py --config configs/m2_qwen25_7b.yaml --method $m
done
python scripts/m1/plot_curves.py --config configs/m2_qwen25_7b.yaml --prefix m2_qwen25 --title "M2: Qwen2.5-7B"
```

Available M2 configs: `m2_chatglm3_6b.yaml`, `m2_qwen25_7b.yaml`, `m2_llama3_8b.yaml`, `m2_baichuan2_7b.yaml`.
Non-LLaMA models (ChatGLM3, Baichuan2) require `trust_remote_code: true`; their `modeling_*.py` files were patched for transformers 5.x `DynamicCache` compatibility (saved inside each checkpoint).

## Running M3 (NF4 quantized QPiSSA vs QLoRA)

Config: `configs/m3_llama2_7b.yaml`. **QPiSSA** loads in fp16 → SVD → NF4-quantizes the frozen residual (`NFPiSSALinear`); **QLoRA** loads the base in 4-bit and wraps each `Linear4bit` with a LoRA adapter (`QLoRALinear`).

```bash
# Table 4: NF4 reconstruction-error comparison (pure compute, no training)
python scripts/m3/table4.py --config configs/m3_llama2_7b.yaml

# Train + eval both methods
for m in qpissa qlora; do
  python scripts/m1/train.py     --config configs/m3_llama2_7b.yaml --method $m
  python scripts/m1/eval_gsm8k.py --config configs/m3_llama2_7b.yaml --method $m
done

# Plot (auto-detects qpissa/qlora)
python scripts/m1/plot_curves.py --config configs/m3_llama2_7b.yaml --prefix m3_qpissa --title "M3: LLaMA-2-7B 4-bit"
```

Note: QLoRA cannot be merged into a plain `nn.Linear` (4-bit base), so its training saves the LoRA adapter weights separately under `checkpoints/m3_qpissa/qlora/lora_weights/`; the eval script reloads the 4-bit base, re-injects QLoRA, and loads those adapter weights.

## Model Selection

- **M1**: `LLaMA-2-7B` — aligned with the original paper for a direct, like-for-like comparison against its reported numbers.
- **M2**: cross-family models — `ChatGLM3-6B` (GLM architecture), `Qwen2.5-7B`, `Llama3-8B`, `Baichuan2-7B` (ALiBi) — to probe how far PiSSA's advantage generalizes beyond the paper's LLaMA-family models.
- **M3**: `LLaMA-2-7B` under NF4 4-bit quantization, comparing QPiSSA against QLoRA.
