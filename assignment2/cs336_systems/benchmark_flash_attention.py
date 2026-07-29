
import triton
import torch
import flashattention
from cs336_basics.model import scaled_dot_product_attention
import csv
import pandas as pd


def generate_inputs(batch_sz, seq_len, d_model, dtype, device="cuda"):
    Q = torch.randn(batch_sz, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(batch_sz, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(batch_sz, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    return Q, K, V


pytorch_fwd_compiled = torch.compile(scaled_dot_product_attention)

def benchmark(batch_sz, seq_len, emb_dim, precision, is_causal, mask):
    print(f"Benchmarking flashattention batch size: {batch_sz}, sequence length: {seq_len}, embedding dimension: {emb_dim}, precision: {precision}, is_causal: {is_causal}")
    
    Q, K, V = generate_inputs(batch_sz, seq_len, d_model=emb_dim, dtype=precision)

    # Warmup
    for _ in range(5):
        _ = pytorch_fwd_compiled(Q, K, V, mask)
        _ = flashattention.FlashAttentionTriton.apply(Q, K, V, is_causal)


    # Pytorch forward
    def pytorch_fwd():
        return pytorch_fwd_compiled(Q, K, V, mask)

    mean_runtime_pytorch_fwd = triton.testing.do_bench(pytorch_fwd)
    print(f"pytorch forward mean time: {mean_runtime_pytorch_fwd}")

    # Triton Flash Attention forward
    def flashattn_triton_fwd():
        return flashattention.FlashAttentionTriton.apply(Q, K, V, is_causal)
    
    mean_runtime_triton_fwd = triton.testing.do_bench(flashattn_triton_fwd)
    print(f"flashattn triton forward mean time: {mean_runtime_triton_fwd}")

    # Pytorch forward + backward
    def pytorch_fwd_bwd():
        out = pytorch_fwd_compiled(Q, K, V, mask)
        loss = out.sum()
        loss.backward()
    
    mean_runtime_pytorch_fwd_bwd = triton.testing.do_bench(pytorch_fwd_bwd, grad_to_none=[Q, K, V])
    print(f"pytorch fwd bwd mean time: {mean_runtime_pytorch_fwd_bwd}")

    # Triton Flash Attention forward + backward
    def flashattn_triton_fwd_bwd():
        out = flashattention.FlashAttentionTriton.apply(Q, K, V, is_causal)
        loss = out.sum()
        loss.backward()
    
    mean_runtime_triton_fwd_bwd = triton.testing.do_bench(flashattn_triton_fwd_bwd, grad_to_none=[Q, K, V])
    print(f"flashattn triton forward backward mean time: {mean_runtime_triton_fwd_bwd}")

    return {
        "seq_len": seq_len,
        "d_model": emb_dim,
        "precision": str(precision).replace("torch.", ""),
        "pytorch_fwd_time_ms": mean_runtime_pytorch_fwd,
        "flash_fwd_time_ms": mean_runtime_triton_fwd,
        "pytorch_fwd_bwd_time_ms": mean_runtime_pytorch_fwd_bwd,
        "flash_fwd_bwd_time_ms": mean_runtime_triton_fwd_bwd,
    }


def main():
    batch_sz = 1
    is_causal = True
    seq_lens = [2**i for i in range(7, 17)]
    emb_dims = [2**i for i in range(4, 8)]
    precisions = [torch.bfloat16, torch.float32]

    output_file = "flashattention_benchmark_results.csv"

    fieldnames = [
        "seq_len",
        "d_model",
        "precision",
        "pytorch_fwd_time_ms",
        "flash_fwd_time_ms",
        "pytorch_fwd_bwd_time_ms",
        "flash_fwd_bwd_time_ms",
        "error",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for seq_len in seq_lens:
        for emb_dim in emb_dims:
            for precision in precisions:

                try:
                    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"))
                    result = benchmark(batch_sz=batch_sz, seq_len=seq_len, emb_dim=emb_dim, precision=precision, is_causal=is_causal, mask=mask)
                    result["error"] = ""

                except torch.cuda.OutOfMemoryError:
                    result = {
                        "seq_len": seq_len,
                        "d_model": emb_dim,
                        "precision": str(precision).replace("torch.", ""),
                        "pytorch_fwd_time_ms": "",
                        "flash_fwd_time_ms": "",
                        "pytorch_fwd_bwd_time_ms": "",
                        "flash_fwd_bwd_time_ms": "",
                        "error": "OOM",
                    }

                    torch.cuda.empty_cache()

                except Exception as e:
                    result = {
                        "seq_len": seq_len,
                        "d_model": emb_dim,
                        "precision": str(precision).replace("torch.", ""),
                        "pytorch_fwd_time_ms": "",
                        "flash_fwd_time_ms": "",
                        "pytorch_fwd_bwd_time_ms": "",
                        "flash_fwd_bwd_time_ms": "",
                        "error": str(e),
                    }

                    torch.cuda.empty_cache()
                
                with open(output_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)
    
    results = pd.read_csv(output_file)

    successful = results["error"].isna() | (results["error"] == "")

    results.loc[successful, "fwd_speedup"] = (
        results.loc[successful, "pytorch_fwd_time_ms"] /
        results.loc[successful, "flash_fwd_time_ms"]
    )

    results.loc[successful, "fwd_bwd_speedup"] = (
        results.loc[successful, "pytorch_fwd_bwd_time_ms"] /
        results.loc[successful, "flash_fwd_bwd_time_ms"]
    )

    results.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()

