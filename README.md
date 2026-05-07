# Spherical KV — Supplementary Code

Code for reproducing the experiments in the paper:
**"Spherical KV: Angle-Domain Attention with Rate-Distortion KV Cache Compression"**

## Setup

```bash
conda create -n sphkv python=3.11 -y
conda activate sphkv
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
huggingface-cli login
```

### requirements.txt

```
transformers>=4.45.0
datasets
scikit-learn
numpy
tqdm
matplotlib
seaborn
```

### Hardware

| Task | Minimum GPU | Recommended |
|------|-------------|-------------|
| Codebook training (1B) | 4 GB VRAM | RTX 3050+ |
| Codebook training (8B+) | 24 GB VRAM | A6000 / A100 |
| W1 experiments (8K ctx) | 24 GB VRAM | A100-80GB |
| W1 experiments (32K+ ctx) | 48 GB VRAM | A100-80GB |
| W2/W3 experiments | 24 GB VRAM | A100-80GB |
| CUDA kernel compilation | CUDA 12.1+ with nvcc | — |

## Reproducing Results

All experiments are run through `experiment_runner.py`. Results are saved to `experiment_results/`.

### Step 0: Train codebooks (one-time per model)

```bash
cd src/codebooks
python generate.py
cd ..
```

Codebooks are saved to the directory specified by `SAVE_DIR` in `src/codebooks/config.py`. Training takes ~2h on A100 (MiniBatchKMeans) or ~13h (full KMeans). Train once, reuse across all experiments.

To change the target model, edit `src/codebooks/config.py`:
```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAVE_DIR   = "codebooks_llama_8b"
```

### Step 1: W1 — Long-context language modeling (Table 4, Figure 4)

PG-19 token-level NLL and perplexity with strided sliding-window protocol.

```bash
python scripts/experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w1 \
  --context_lengths 8192 32768 \
  --modes dense sphkv sphkv_recon sphkv_angle sphkv_rd \
  --budgets 48 56 64 80 96 112 128 160 \
  --n_warm 8 --n_meas 64 --n_trials 3 \
  --device cuda
```

### Step 2: W2 — Retrieval QA (Table 5, Figure 5 A4)

Multi-hop QA on HotpotQA and 2WikiMultiHopQA with distractor and position sweeps.

```bash
python scripts/experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w2 \
  --modes dense sphkv \
  --budgets 60 \
  --w2_task hotpotqa \
  --w2_max_samples 50 \
  --w2_distractors 0 3 5 \
  --w2_positions early middle late \
  --device cuda
```

Repeat with `--w2_task 2wikimqa` for the second dataset.

### Step 3: W3 — Agentic rollouts (Table 6, Figure 5 A5)

Multi-step tool-use trajectories measuring behavioral divergence.

```bash
python scripts/experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w3 \
  --modes dense sphkv \
  --budgets 60 \
  --w3_source toolbench \
  --w3_max_episodes 30 \
  --w3_max_steps 10 \
  --w3_seeds 3 \
  --device cuda
```

### Step 4: Ablations (Figure 5 A0-A3)

```bash
python scripts/experiment_runner.py \
  --models meta-llama/Llama-3.1-8B-Instruct \
  --codebook_dirs codebooks/codebooks_llama_8b \
  --workloads w1 \
  --context_lengths 8192 32768 \
  --modes dense sphkv sphkv_recon sphkv_angle sphkv_rd \
         keepdrop quant_only decoupled \
  --budgets 48 64 80 112 160 \
  --device cuda
```

### HBM traffic measurement (Table A.2)

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    --nvtx --nvtx-include angle_logits \
    --csv --log-file ncu_report.csv \
    python scripts/experiment_runner.py --modes sphkv --budgets 60
```

## File Structure

```
├── README.md
├── requirements.txt
├── run.sh                                Convenience launcher for W1/W2/W3
├── .gitignore
│
├── configs/
│   └── config.py                         Pipeline config: tiers, budget, model name
│
├── scripts/
│   └── experiment_runner.py              Full experiment harness for W1/W2/W3
│
├── docs/
│   └── tier_allocation_dashboard.html    Pre-rendered interactive dashboard
│
└── src/
    ├── _path_setup.py                    Adds src/ + configs/ to sys.path
    │
    ├── spherical_kv_pipeline.py          Main pipeline: ADA attention + RDR allocation
    ├── decode_kernel.cu                  Fused CUDA decode kernel (per-position Q back-rotation)
    ├── fused_decode.cpp                  C++ binding for the CUDA kernel
    ├── fused_decode.py                   Python wrapper for the kernel
    ├── sphkv_lut.py                      Per-layer page pool, kernel dispatch
    │
    ├── allocation.py                     Greedy knapsack tier allocator (Algorithm 1)
    ├── distortion_proxy.py               Calibrated distortion proxy (Appendix C)
    ├── calibrate_lambda.py               Offline calibration of tier lambdas
    ├── calibrate_lambda_codebook.py      Codebook-aware lambda calibration
    │
    ├── tiers.py                          Tier dataclass and builder
    ├── token_state.py                    Per-token state tracking
    ├── paging.py                         Page layout and bitpacking
    ├── pagebuilder.py                    Page construction from quantized codes
    ├── pointer_table.py                  Page table for kernel dispatch
    ├── bitpacking.py                     Bit-level packing utilities
    │
    ├── codebook_loader.py                Load trained codebooks from disk
    ├── quantization.py                   Spherical quantization (radius + angular codes)
    ├── spherical_parameterization.py     K -> (r, theta) decomposition per group
    │
    ├── llama_hooks.py                    Pre-RoPE K capture + patched decode forward
    ├── resuse_proxy.py                   Token reuse proxy (EMA attention weights)
    ├── stability_proxy.py                Logit stability proxy for drift detection
    │
    ├── evaluate.py                       Standalone evaluation with strided perplexity
    ├── dataset_w2.py                     HotpotQA / 2WikiMultiHopQA loaders
    ├── dataset_w3.py                     Agentic tool-use task loaders
    ├── baselines.py                      Baseline KV-compression methods
    ├── ablation_modes.py                 Ablation harness for paper figure 5
    ├── negative_controls.py              Sanity-check baselines
    ├── results.py                        Result aggregation and figures
    ├── visualize.py                      Interactive tier allocation dashboard generator
    ├── paper_plots.py                    Paper-quality plot rendering
    ├── analysis.py                       Post-hoc analysis helpers
    ├── hardware_audit.py                 GPU/CPU/memory environment table
    ├── vllm_backend.py                   Optional vLLM backend
    │
    └── codebooks/                        Codebook training (run once per model)
        ├── config.py                     Training config: K-means mode, samples, chunk size
        ├── generate.py                   Codebook training (pre-RoPE K-means on C4)
        ├── generate_importance.py        Variant: importance-weighted training
        └── dataset_loader.py             C4 data loading for codebook training
```

## Configuration

There are two config files serving different purposes.

### `configs/config.py` — Pipeline and experiment config

| Parameter | Description |
|-----------|-------------|
| `MODEL_NAME` | HuggingFace model identifier |
| `DEVICE` | `cuda` or `cpu` |
| `TIERS` | List of (tier_id, name, group_size, b_theta, K_centroids) |

Tier definitions follow the paper (Section 2.2):

| Tier | Name | Group size (g) | b_theta | Centroids (K) |
|------|------|----------------|---------|----------------|
| 1 | High | 16 | 6 | 64 |
| 2 | Mid | 16 | 4 | 16 |
| 3 | Low | 32 | 3 | 8 |

### `src/codebooks/config.py` — Codebook training config

| Parameter | Description |
|-----------|-------------|
| `KMEANS_MODE` | `"full"` (best quality, slow) or `"minibatch"` (fast) |
| `NUM_SAMPLES` | Number of C4 texts for training (default: 2000) |
| `SEQ_LEN` | Max sequence length per sample (default: 512) |
| `MAX_VECS_PER_GROUP` | Cap on training vectors per codebook (default: 100,000) |
| `SAVE_DIR` | Output directory for trained `.pt` files |

To train codebooks for a new model, edit `src/codebooks/config.py` and run:
```bash
cd src/codebooks
python generate.py
```

## Expected Results

Results to be filled after experiments on target hardware.

### W1: Language Modeling (PG-19)

| Model | Context | Dense PPL | SphKV PPL | PPL ratio | Speedup | KV Reduction |
|-------|---------|-----------|-----------|-----------|---------|--------------|
| | 8K | | | | | |
| | 32K | | | | | |
| | 128K | | | | | |

### W2: Retrieval QA (HotpotQA)

| Model | Distractors | Dense EM | SphKV EM | Dense F1 | SphKV F1 |
|-------|-------------|----------|----------|----------|----------|
| | 0 | | | | |
| | 3 | | | | |
| | 5 | | | | |

### W3: Agentic Rollouts

| Model | Dense success | SphKV success | Disagree rate |
|-------|--------------|---------------|---------------|
| | | | |

## License

This code is provided for review purposes only.