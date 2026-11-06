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
| **M0** | Initialization correctness checks, controlled PEFT alignment, paired MNIST mechanism experiment, and weight/residual spectra and distributions; rerun the revised protocol before making new result claims | Local / free |
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

**Table 4 — NF4 quantization reconstruction error**:

| | Baseline NF4 error | PiSSA-residual NF4 error | Improvement |
|---|:---:|:---:|:---:|
| value (Frobenius, 2 layers) | 7.19% | 1.51% | **79.0%** |

> Note: the number above uses the **Frobenius** norm over only 2 sampled layers, so the reduction ratio is inflated. `src/scripts/m3/table4.py` has since been updated to match the original paper's metric — **nuclear-norm** error-reduction ratio, computed over **all** layers and grouped by projection type (Q/K/V/O/Gate/Up/Down), with `AVG` = mean of per-type means. Re-run on the server (`python src/scripts/m3/table4.py --config configs/m3_llama2_7b.yaml`) to obtain the paper-aligned table (expected PiSSA AVG ≈ 19% for LLaMA-2-7B, highest on Q). The Frobenius reduction is still emitted as a reference field.

> QPiSSA mathematically reduces the NF4 quantization error of the frozen residual (Table 4), but in end-to-end training it underperformed QLoRA. The most likely cause is our from-scratch NF4 path in `src/pissa/layer_nf4.py` (CPU→GPU dequantize each forward) being numerically weaker than bitsandbytes' native fused 4-bit matmul kernel used by QLoRA. Reported honestly as an engineering limitation.

## Directory Layout

```
src/pissa/          From-scratch implementation: svd_init / layer (PiSSALinear) / layer_nf4 (NFPiSSALinear) / inject
src/scripts/       Experiment entry points and shared model_ops.py / data_ops.py
src/lora/          LoRA layers and injection; quantized branches remain separate
src/utils.py       Shared configuration, seed, path, and dtype helpers
                   M2/M3 reuse src/scripts/m1/ with their respective configs
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
> grep -vE '^(torch|torchvision|triton)==' requirements.txt > /tmp/req_no_torch.txt
> pip install -r /tmp/req_no_torch.txt     # all other packages are pinned to exact versions
> python -c "import torch;print(torch.__version__)"   # must still print 2.8.0+cu128
> ```
>
> The environment can be configured in CPU-only ("no-GPU") mode; a GPU is only needed at training time.

## Running M0

```bash
# 1. Mathematical checks + Fast quality + exact/Fast PEFT and LoRA alignment
python src/scripts/m0/verify_init.py --in-features 1024 --out-features 1024 --r 64 --seeds 0 1 2 --niters 2 4 --device cpu

# 2. Paired mechanism experiment: retrain one shared base per seed
python src/scripts/m0/mnist_toy.py --epochs 3 --ft-steps 300 --seeds 0 1 2 --eval-every 100 --device cuda

# 3a. Paper-sample path: set this to an existing local safetensors directory
python src/scripts/m0/plot_singular.py --local-path "PATH_TO_LLAMA2_SNAPSHOT" --layer model.layers.0.self_attn.q_proj --r 128

# 3b. Alternative real-weight demonstration (network required; not the paper sample)
python src/scripts/m0/plot_singular.py --source auto
```

Use the project's Python environment (PEFT 0.18.1); on this Windows checkout it is
`D:\app\anaconda\envs\PiSSA\python.exe`. `--device cpu` also supports the toy run;
CUDA is optional. No LLM is downloaded for PEFT checks: a tiny random Llama is
constructed locally. The toy run downloads official MNIST if it is absent from
`data/mnist`; this cache does not move with `--output`.

All three scripts interpret `--output DIR` as an experiment root containing
`metrics/` and `figures/`. The default remains `results/m0/exp1/` and overwrites
its previous results. Use a different output directory for smoke checks.
Reports start as `running`, update incrementally, and use same-directory atomic
JSON replacement. Exit codes: `0` complete/pass, `1` failure/error, `2` partial
(including `--skip-peft`), and `64` invalid arguments. A leftover `running` report
means incomplete, not a diagnosed cause. Saving failures are errors, not passes.

The revised protocol distinguishes arithmetic preservation from principal
selection using known gapped spectra, independent FP64 references, and
deliberately incorrect test cases. Fast SVD uses predeclared approximation
safety bounds (residual Frobenius ratio <= 1.5 where the optimal tail is nonzero)
and reports actual ratios, not a universal near-optimality guarantee. Near-zero
tails use scale-aware absolute bounds; near-degenerate directions are diagnostic,
not an automatic failure. The shared initializer still computes in FP32, and its
default Fast solver dimension remains `min(k, r+8)`; reference alignment explicitly
uses `q=r`. This implementation currently accepts only `alpha=r` for PiSSA; this
is an implementation restriction, not a mathematical limitation of PiSSA.

MNIST uses three Linear layers and two hidden ReLUs. Each seed retrains a base
and compares `fc_full` (fc weight+bias), `full_ft` (all model weight/bias), LoRA,
and PiSSA (fc adapters only, frozen bias), with paired fine-tuning batches.
LoRA A now uses Gaussian std `1/r`, B zero. Historical single-seed results used
a different initialization/data-order protocol and must not be mixed with new
results. The official even-class test split is evaluated at predeclared steps;
it must not select hyperparameters, seeds, or checkpoints. Per-seed metrics and
paired differences are primary; n=3 summaries are descriptive, not significance
claims. This is a mechanism experiment, not a reproduction of Fig.2(a)'s 2D path.

The spectrum script plots W/residual spectra and element distributions with
shared bins (Fig.3(a)(b)(c)(f)-style, no quantization). Real-source or target-key
failures stop by default. `--random` is an explicit synthetic demonstration;
`--allow-random-fallback` explicitly permits a labeled fallback after remote
failure. Local loads never silently switch sources. Network reads require valid
HTTP 206 byte ranges, bounded headers, consistent available ETags, and finite
retries. Reports retain actual source/key/shard/original dtype and HTTP metadata.

Run offline CPU tests with `python -m pytest tests -q`. PEFT tests skip explicitly
when imports are missing/incompatible. Skipped tests are not alignment evidence.
Seeds do not guarantee bitwise equality across hardware or dependency versions.

**Revised M0 status:** implementation smoke results do not replace a full run of
this protocol. No PiSSA performance ordering is a pass criterion. JSON schema v2
stores per-seed toy results under `results.runs`; do not run the old
`src/scripts/plot/regenerate_figures.py` on it. Its compatibility and the paper draft
are deferred; this stage does not rewrite M1–M3 figures or `paper/figures/`.

## Running M1

LLaMA-2-7B on MetaMathQA, evaluated on GSM8K. Config: `configs/m1_llama2_7b.yaml` (rank `r=128`, paper-aligned). Methods: our **PiSSA** / **LoRA** / **Full-FT**.

```bash
# 0. Prepare data (downloads MetaMathQA + GSM8K via ModelScope, writes local jsonl)
python src/scripts/m1/prepare_data.py --config configs/m1_llama2_7b.yaml          # add --limit 2000 for a smoke run

# 1. Train each method (saves merged model to checkpoints/m1/<method>)
python src/scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method pissa  # add --limit 64 for a smoke run
python src/scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method lora
python src/scripts/m1/train.py --config configs/m1_llama2_7b.yaml --method full

# 2. Evaluate on GSM8K
python src/scripts/m1/eval_gsm8k.py --config configs/m1_llama2_7b.yaml --method pissa

# 3. Plot loss curves + accuracy bars
python src/scripts/m1/plot_curves.py --config configs/m1_llama2_7b.yaml
```

Memory strategy on a single 96GB GPU: PiSSA/LoRA train in fp32 (frozen base + tiny adapter) with gradient checkpointing on — fp32 activations are large, so disabling it OOMs even on 96GB; Full-FT cannot fit 7B in fp32 (params + AdamW state + grads ≈ 112GB), so it uses bf16 + 8-bit AdamW + gradient checkpointing (`methods.full` in the config). TF32 matmul is enabled in `train.py` to speed up the fp32 paths with negligible numerical impact (this is the main speed lever; it saves no memory). The smoke path (small `--limit`) is meant to be run first to validate the pipeline before launching the long runs.

## Running M2 (cross-model generality)

M2 reuses the M1 scripts with per-model configs. Each model has its own `configs/m2_<model>.yaml`.

```bash
for m in pissa lora; do
  python src/scripts/m1/train.py     --config configs/m2_qwen25_7b.yaml --method $m
  python src/scripts/m1/eval_gsm8k.py --config configs/m2_qwen25_7b.yaml --method $m
done
python src/scripts/m1/plot_curves.py --config configs/m2_qwen25_7b.yaml --prefix m2_qwen25 --title "M2: Qwen2.5-7B"
```

Available M2 configs: `m2_chatglm3_6b.yaml`, `m2_qwen25_7b.yaml`, `m2_llama3_8b.yaml`, `m2_baichuan2_7b.yaml`.
Non-LLaMA models (ChatGLM3, Baichuan2) require `trust_remote_code: true`; their `modeling_*.py` files were patched for transformers 5.x `DynamicCache` compatibility (saved inside each checkpoint).

## Running M3 (NF4 quantized QPiSSA vs QLoRA)

Config: `configs/m3_llama2_7b.yaml`. **QPiSSA** loads in fp16 → SVD → NF4-quantizes the frozen residual (`NFPiSSALinear`); **QLoRA** loads the base in 4-bit and wraps each `Linear4bit` with a LoRA adapter (`QLoRALinear`).

```bash
# Table 4: NF4 reconstruction-error comparison (pure compute, no training)
python src/scripts/m3/table4.py --config configs/m3_llama2_7b.yaml

# Train + eval both methods
for m in qpissa qlora; do
  python src/scripts/m1/train.py     --config configs/m3_llama2_7b.yaml --method $m
  python src/scripts/m1/eval_gsm8k.py --config configs/m3_llama2_7b.yaml --method $m
done

# Plot (auto-detects qpissa/qlora)
python src/scripts/m1/plot_curves.py --config configs/m3_llama2_7b.yaml --prefix m3_qpissa --title "M3: LLaMA-2-7B 4-bit"
```

Note: QLoRA cannot be merged into a plain `nn.Linear` (4-bit base), so its training saves the LoRA adapter weights separately under `checkpoints/m3_qpissa/qlora/lora_weights/`; the eval script reloads the 4-bit base, re-injects QLoRA, and loads those adapter weights.

## Model Selection

- **M1**: `LLaMA-2-7B` — aligned with the original paper for a direct, like-for-like comparison against its reported numbers.
- **M2**: cross-family models — `ChatGLM3-6B` (GLM architecture), `Qwen2.5-7B`, `Llama3-8B`, `Baichuan2-7B` (ALiBi) — to probe how far PiSSA's advantage generalizes beyond the paper's LLaMA-family models.
- **M3**: `LLaMA-2-7B` under NF4 4-bit quantization, comparing QPiSSA against QLoRA.
