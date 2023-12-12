import torch
import torch.utils.benchmark as benchmark


def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Input for benchmarking
x = torch.randn(1000000, 256)


# Ensure that both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from pytorchBench import batched_dot_mul_sum',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using mul and sum')

# Can use AMX
t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from pytorchBench import batched_dot_bmm',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using bmm')

def bench():

    print('Pytorchbench:')
    # Baseline, prints average time per iteration
    print(t0.timeit(100))
    # Optimized bmm, prints average time per iteration
    print(t1.timeit(100))

