# CS336: Language Modeling from Scratch

My implementation and experiments for [Stanford CS336 — Language Modeling from Scratch](https://cs336.stanford.edu/) (Assignments 1–3). Everything here — the tokenizer, the transformer, the training loop, the distributed systems code, and the scaling-law tooling — is built from first principles, mostly in raw PyTorch and Triton without relying on Hugging Face or other high-level libraries.

Full writeups with results and analysis are in [`writeup.md`](./writeup.md).

## Highlights

**Assignment 1 — Basics**
- Byte-pair encoding tokenizer implemented from scratch (trained on TinyStories and OpenWebText), including the fast merge-counting logic used during training
- Transformer language model built from scratch: multi-head self-attention, RoPE, SwiGLU feed-forward, RMSNorm
- AdamW optimizer implemented from scratch, with cosine learning-rate schedule and gradient clipping
- Learning-rate and batch-size ablations, with FLOPs/parameter-count accounting for GPT-2-scale models
- End-to-end training + checkpointing + autoregressive text generation on TinyStories

**Assignment 2 — Systems**
- Custom FlashAttention forward and backward kernels written in Triton, up to ~10x faster than PyTorch SDPA at small sequence lengths, benchmarked up to 65K tokens on an H100
- Mixed-precision (fp16/bf16) training analysis — where PyTorch autocast keeps ops in fp32 and why
- Memory profiling of forward vs. full training steps across context lengths
- Distributed training from scratch: naive DDP, flattened-gradient DDP, and DDP with overlapped backward/communication (cut gradient-sync overhead from ~54% to ~3% of step time)
- Optimizer state sharding (ZeRO stage 1-style), reducing peak memory after the optimizer step by ~30% on 2× GPUs
- `torch.compile` benchmarking across model/context sizes

**Assignment 3 — Scaling Laws**
- Chinchilla IsoFLOPs analysis to derive compute-optimal model/dataset size tradeoffs, fitting power laws to loss vs. compute
- Scaling-law experiment design under a fixed compute budget, allocating denser coverage to cheaper runs and using larger runs to validate the fitted curve

## Repo structure

```
.
├── writeup.md              # combined writeup for all three assignments
├── assignment1/             # tokenizer, transformer, optimizer, training loop
│   ├── src/                 # BPE tokenizer, model, optimizer, training loop
│   ├── tests/
│   └── plots/                # LR / batch-size ablation plots
├── assignment2/              # systems: benchmarking, mixed precision, flash attention, distributed training
│   ├── cs336_systems/          # benchmarking, DDP, optimizer sharding, flash attention (triton)
│   ├── cs336-basics/           # assignment 1 model code, reused as a dependency
│   ├── results/                # benchmark CSVs and profiling plots
│   └── tests/
└── assignment3/              # scaling laws
    ├── cs336_scaling/           # chinchilla_isoflops.py
    ├── scripts/
    ├── examples/
    └── tests/
```

Each assignment retains its original git history (imported via `git subtree`) — see the commit log for the incremental development of each component.

## Stack

PyTorch, Triton, einops, pytest.
