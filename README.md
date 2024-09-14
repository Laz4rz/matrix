# matrix

Picked the mantle to implement matrices and matmul in Python, C, and CUDA. The inspiration is [here](https://github.com/spikedoanz/matmul). 

A [gem of knowledge](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf), or where does these GEMM something names come from?
![alt text](images/matmul_naming.png)


## Performance

#### Baseline 
- Numpy (multithreaded, M2 Pro): 
```
(128, 128, 128): 57.52 +/- 28.71 GFLOPS
(512, 512, 512): 253.81 +/- 25.85 GFLOPS
(1024, 1024, 1024): 274.68 +/- 56.40 GFLOPS
```
- Numpy (single thread, M2 Pro):
```
(128, 128, 128): 92.89 +/- 9.58 GFLOPS
(512, 512, 512): 100.50 +/- 8.87 GFLOPS
(1024, 1024, 1024): 104.77 +/- 2.32 GFLOPS
```

#### Naive
- Python (matrix.py): ~0.0004 GFLOPS (Ryzen 3600), ~0.0010 (M2 Pro)
- C      (matrix.c): 
  - M2 Pro: ~0.31 GFLOPS (-O0), higher optimization breaks benchmarking
  - Ryzen 3600: ~0.13 (-O0), ~0.55 (-O1), ~1.00-1.6 for small and 0.33 for large (-O2), ~1.9 small, 0.9 medium, 0.33 for large (-O3)

## Python

Matrices are implemented using 3D row-major strided representation. To create a Matrix object:

```python
m = Matrix(d=2, h=2, w=2, data=[i for i in range(8)])
```

Above creates a 2x2x2 matrix with 0..8 values as entries. Operations on matrices work as Python usual, `@` for matmul.

Testing Python matrix and matmul implementation:
```
python -m unittest tests.test_matrix.TestMatrix
```

## C

Same as python, 3D row-major strided representation matrices. Matrix is built with a `Matrix` struct. To access, set and matmul use: `get`, `set`, and `matmul`. To create a Matrix:

```C
Matrix A;
Matrix B;
Matrix C;

allocate_matrix_random(&A, 1, M, N);
allocate_matrix_consecutive(&B, 1, N, K);
allocate_matrix_zeros(&C, 1, M, K);
```

## Strassens Algorithm for Matrix Multiplication (or how to cheat complexity with clever algebra)

So the typical time complexity of matrix multiplication is $O(n^3)$, divide and conquer than Strassens is based on is also $O(n^3)$. What is the catch then? Turns out that we can be smart about the algebraic computations, and reduce the number of recursive calls from 8 to 7, bringing the time complexity down to $O(n^(2.81))$.

![alt text](images/strassens.png)

## Scratch

- great read: https://salykova.github.io/matmul-cpu
- obvious read: https://siboehm.com/articles/22/CUDA-MMM
- Efficient matmul on CPU: https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf
- SIMD intro: https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
- ARM instructions (suppossedly M series too): https://developer.arm.com/documentation/dui0801/l/A64-SIMD-Vector-Instructions/A64-SIMD-Vector-instructions-in-alphabetical-order
