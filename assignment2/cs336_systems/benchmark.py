
import argparse
from cs336_basics.model import BasicsTransformerLM
import torch
import timeit
import statistics
from contextlib import nullcontext
import torch._functorch.config


def benchmark_model(
    model_params,
    input_data,
    num_warmups,
    num_trials,
    mode,
    use_mixed_precision,
    profile_memory,
    compile,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BasicsTransformerLM(**model_params)
    model.to(device)

    if compile:
        print("JIT compiled")
        torch._functorch.config.donated_buffer = False
        model = torch.compile(model)

    input_data = input_data.to(device)

    optimizer = torch.optim.AdamW(model.parameters())

    mp_dtype = torch.bfloat16

    if use_mixed_precision:
        cast_context = torch.autocast(device_type=device, dtype=mp_dtype)
    else:
        cast_context = nullcontext()


    def execute_step():
        with cast_context:
            output = model(input_data).mean()
        
        if mode == "backward":
            with cast_context:
                optimizer.zero_grad(set_to_none=True)
                output.backward()

        elif mode == "train":
            with cast_context:
                optimizer.zero_grad(set_to_none=True)
                output.backward()
                optimizer.step()


    for _ in range(num_warmups):
        execute_step()

    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    times = []
    for _ in range(num_trials):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = timeit.default_timer()
        
        execute_step()

        if device == "cuda":
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    if profile_memory:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    # return the average and standard deviation of times
    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_warmups", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["forward", "backward", "train"],
        default="forward",
    )
    parser.add_argument("--use_mixed_precision", action='store_true')
    parser.add_argument("--profile_memory", action='store_true')
    parser.add_argument("--compile", action="store_true")

    args = parser.parse_args()
    print(args)

    model_params = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }

    input_data = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
    )

    res = benchmark_model(
        model_params,
        input_data,
        args.num_warmups,
        args.num_trials,
        args.mode,
        args.use_mixed_precision,
        args.profile_memory,
        args.compile,
    )
    print(f"Mean time per step: {res['mean']:.6f} s")
    print(f"Std: {res['std']:.6f} s")


if __name__ == "__main__":
    main()

