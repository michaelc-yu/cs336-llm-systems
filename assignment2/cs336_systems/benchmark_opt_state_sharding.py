import torch
import os
import timeit
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.optimizer_state_sharding import OptimizerStateSharding


NUM_STEPS = 1

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def benchmark_opt_state_sharding(rank, world_size, model_params, input_data, backend, shard_optimizer: bool):

    setup(rank, world_size, backend)

    use_gpu = torch.cuda.is_available() and backend == 'nccl'

    if use_gpu:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = "cpu"

    torch.cuda.reset_peak_memory_stats(device=None)

    model = BasicsTransformerLM(**model_params)
    model.to(device)

    peak_mem_after_model_init = torch.cuda.max_memory_allocated(device=None)
    print(f"[Rank {rank}] Peak memory after model init: {peak_mem_after_model_init}")

    if shard_optimizer:
        optimizer = OptimizerStateSharding(model.parameters(), torch.optim.AdamW)
    else:
        optimizer = torch.optim.AdamW(model.parameters())


    if use_gpu:
        input_data = input_data.to(device)
    

    start_time = timeit.default_timer()

    for _ in range(NUM_STEPS):

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats(device=None)

        output = model(input_data).mean()
        output.backward()

        peak_mem_before_optimizer_step = torch.cuda.max_memory_allocated(device=None)
        print(f"[Rank {rank}] Peak memory before optimizer step: {peak_mem_before_optimizer_step}")

        if use_gpu:
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device=None)

        optimizer.step()

        peak_mem_after_optimizer_step = torch.cuda.max_memory_allocated(device=None)
        print(f"[Rank {rank}] Peak memory after optimizer step: {peak_mem_after_optimizer_step}")

    end_time = timeit.default_timer()
    total_time = end_time - start_time
    print(f"Total training time for {NUM_STEPS} steps: {total_time}")


    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_sz", type=int, default=4)
    parser.add_argument("--backend", type=str, default='nccl')
    parser.add_argument("--shard_optimizer", action='store_true')

    args = parser.parse_args()

    assert args.batch_sz % args.world_size == 0

    model_params = {
        "vocab_size": 10000,
        "context_length": 128,
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,
        "rope_theta": 10000,
    }

    input_data = torch.randint(
        low=0,
        high=10000,
        size=(args.batch_sz, 128),
    )

    mp.spawn(fn=benchmark_opt_state_sharding, args=(args.world_size, model_params, input_data, args.backend, args.shard_optimizer), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
