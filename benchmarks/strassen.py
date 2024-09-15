import numpy as np
import time

import sys
sys.path.append("..")
from matrix import matrix
from strassens import strassens


def benchmark_matmul(M, N, K, num_iterations):
    np.random.seed(0)
    A = np.random.rand(M, K).astype(np.float32).flatten()
    B = np.random.rand(K, N).astype(np.float32).flatten()
    A = matrix.Matrix(1, M, K, list(A))
    B = matrix.Matrix(1, K, N, list(B))

    for _ in range(num_iterations):
        start_time = time.perf_counter()

        C = A @ B

        end_time = time.perf_counter()

        iteration_time = end_time - start_time
        flops = 2.0 * M * N * K
        flops_per_second = flops / iteration_time / 1e9
        print(f"   {flops_per_second:.6f} GFLOPS")

    return flops_per_second

def main():
    sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)]
    num_iterations = 5

    max_flops = 0
    max_config = None

    results = {}
    for i in range(1, 2):
        print(f"iteration {i}")
        for M, N, K in sizes:
            print(f"   Running benchmark for M={M}, N={N}, K={K}")
            flops = benchmark_matmul(M, N, K, num_iterations)
            if flops > max_flops:
                max_flops = flops
                max_config = (M, N, K)
            if (M, N, K) in results:
                results[(M, N, K)].append(flops)
            else:
                results[(M, N, K)] = [flops]

    print(f"\nAverage FLOPS | (1x100 iterations):")
    for config, flops in results.items():
        avg_flops = sum(flops) / len(flops)
        std_flops = np.std(flops)
        print(f"{config}: {avg_flops:.2f} +/- {std_flops:.2f} GFLOPS")


    print(f"\nConfiguration with highest FLOPS: {max_config}")
    print(f"Highest FLOPS: {max_flops:.2f} GFLOPS")

if __name__ == "__main__":
    main()
