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
| **M1** | Single model **LLaMA-2-7B** (paper-aligned), PiSSA vs LoRA (±Full-FT), proving faster/better convergence (MetaMathQA → GSM8K), validated against the paper's numbers | RTX PRO 6000 96GB |
| **M2** | Repeat on **newer SOTA models** (if M1 matches the paper) to demonstrate generality on modern LLMs — more convincing than reusing the paper's models (Table 2 spirit) | RTX PRO 6000 96GB |
| **M3** | Quantized QPiSSA: error-reduction table (Table 4, pure compute) + 4-bit training convergence (Fig.5) | Single GPU |

## Directory Layout

```
src/pissa/          From-scratch implementation: svd_init / layer (PiSSALinear) / inject
scripts/m0..m3/     Runnable scripts per milestone
configs/            Experiment configs (base.yaml holds shared hyperparameters)
results/m*/exp*/    Experiment artifacts (figures, metrics JSON)
.log/               Local-only project records (log.md / HANDOFF.md); git-ignored, not pushed
data/ checkpoints/  Large files, server-only, git-ignored
```

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

## Model Selection

- **M1**: `LLaMA-2-7B` — aligned with the original paper for a direct, like-for-like comparison against its reported numbers.
- **M2**: newer SOTA models (e.g. `Llama-3.1-8B` / `Qwen3-8B` / `Gemma-2-9B`), conditional on M1 matching the paper — showing PiSSA still holds on a new generation of LLMs, which is more compelling than repeating the paper's exact models.
