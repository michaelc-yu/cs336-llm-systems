import argparse
from torch import nn
import torch
from einops import einsum
import math
import timeit
import csv
from cs336_basics.model import scaled_dot_product_attention


def benchmark(batch_size, seq_len, d_model, device, num_warmup, num_forward, num_backward, compile):
    """
    Benchmark time and memory usage for forward and backward passes attention implementation.
    Returns total forward time, total backward time, and the total memory allocated (MB) after forward passes.
    """
    print(f"Benchmark d_model: {d_model}, seq_len: {seq_len}")
    shape = (batch_size, seq_len, d_model)

    Q = torch.rand(shape, requires_grad=True, device=device)
    K = torch.rand(shape, requires_grad=True, device=device)
    V = torch.rand(shape, requires_grad=True, device=device)

    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()

    if compile:
        print("JIT compiled")
        torch._functorch.config.donated_buffer = False
        attn_func = torch.compile(scaled_dot_product_attention)
    else:
        attn_func = scaled_dot_product_attention

    # warmup
    for _ in range(num_warmup):
        out = attn_func(Q, K, V, mask)
        loss = out.sum()
        loss.backward()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # start 100 forward passes
    before_fwd_time = timeit.default_timer()
    for _ in range(num_forward):
        out = attn_func(Q, K, V, mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
    after_fwd_time = timeit.default_timer()
    fwd_time = after_fwd_time - before_fwd_time
    print(f"Forward time: {fwd_time:.4f}s")

    if torch.cuda.is_available():
        mem_allocated_in_MB = torch.cuda.memory_allocated() / (1024**2)
        print(f"CUDA memory usage after forward: {mem_allocated_in_MB:.4f}MB")

    loss = out.sum()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 100 backward passes
    before_bwd_time = timeit.default_timer()
    for _ in range(num_backward):
        loss.backward(retain_graph=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    after_bwd_time = timeit.default_timer()
    bwd_time = after_bwd_time - before_bwd_time
    print(f"Backward time: {bwd_time:.4f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return fwd_time, bwd_time, mem_allocated_in_MB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    num_warmup = 10
    num_forward = 100
    num_backward = 100
    results = []

    for d_model in d_models:
        for seq_len in seq_lens:
            try:
                total_fwd_time, total_bwd_time, mem_allocated_in_MB = benchmark(batch_size, seq_len, d_model, device, num_warmup, num_forward, num_backward, args.compile)
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "total_fwd_time": total_fwd_time,
                    "total_bwd_time": total_bwd_time,
                    "total_mem_allocated_in_MB_after_forward": mem_allocated_in_MB,
                    "JIT_compiled": args.compile,
                    "exit_error": 0,
                })
            except:
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "exit_error": "OOM",
                })
                # clean up any leftover CUDA memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    out_file = "benchmark_attention_results.csv" if not args.compile else "benchmark_attention_results_compiled.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "seq_len",
                "d_model",
                "total_fwd_time",
                "total_bwd_time",
                "total_mem_allocated_in_MB_after_forward",
                "exit_error",
                "JIT_compiled",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
