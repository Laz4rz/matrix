# matrix

Picked the mantle to implement matrices and matmul in Python, C, and CUDA. The inspiration is [here](https://github.com/spikedoanz/matmul). 

## Python

Matrices are implemented using 3D row-major strided representation. To create a Matrix object:

```python
m = Matrix(d=2, h=2, w=2, data=[i for i in range(8)])
```

Above creates a 2x2x2 matrix with 0..8 values as entries. Operations on matrices work as Python usual, `@` for matmul.

Python matrix and matmul implementation:
```
python -m unittest tests.test_matrix.TestMatrix
```

