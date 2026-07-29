
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit


NUM_WARMUPS = 5
NUM_STEPS = 10

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def all_reduce_benchmark(rank, world_size, data_size, backend):

    num_elements = (data_size * 1024**2) // 4
    setup(rank, world_size, backend)
    data = torch.randn(num_elements)

    use_gpu = torch.cuda.is_available() and backend == 'nccl'

    if use_gpu:
        torch.cuda.set_device(rank)
        data = data.to(f'cuda:{rank}')

    for _ in range(NUM_WARMUPS):
        dist.all_reduce(data, async_op=False)

        if use_gpu:
            torch.cuda.synchronize()
    
    start = timeit.default_timer()
    for _ in range(NUM_STEPS):
        dist.all_reduce(data, async_op=False)

        if use_gpu:
            torch.cuda.synchronize()
    
    end = timeit.default_timer()
    elapsed_time = end - start
    if rank == 0:
        print(f"Data size (MB): {data_size}, Num process: {world_size}, Time (s): {elapsed_time}")

    dist.destroy_process_group()


def benchmark(backend):
    data_sizes = [1, 10, 100, 1024] # num MB
    num_processes = [2, 4, 6]
    for d in data_sizes:
        for num_process in num_processes:
            mp.spawn(fn=all_reduce_benchmark, args=(num_process, d, backend), nprocs=num_process, join=True)


if __name__ == "__main__":
    backend = "gloo"
    benchmark(backend)
