import numpy as np

class Matrix:
    # 3d implemented with strided representation in row-major order
    # matrices are treated as depth-number of stacked 2d matrices
    def __init__(self, d=1, h=1, w=1, data=None, verbose=False):
        self.shape = (d, h, w)
        self.size = d * h * w
        self.stride = (h * w, w, 1)
        if data is not None:
            assert len(data) == self.size
            self.data = data
        else:
            self.data = [0 for _ in range(h) for _ in range(d) for _ in range(w)]
            # self.data = [1 ,2 ,3 ,4]
        self.verbose = verbose


    def __getitem__(self, idx):
        if self.verbose: print(f"__getitem__ idx={idx} for {self.__repr__()}")
        idx = list(idx) if isinstance(idx, tuple) else [idx]
        idx = list(idx) + [slice(None, None) for _ in range(3-len(idx))]

        for i in range(len(idx)):
            if isinstance(idx[i], slice):
                start, stop = idx[i].start, idx[i].stop
                start = 0 if start is None else start
                stop = self.shape[i] if stop is None else stop
                idx[i] = (start, stop)
            else:
                idx[i] = (idx[i], idx[i] + 1)
        if self.verbose: print(f"idx: {idx}")
            
        temp = []
        if self.verbose: print("stride:", self.stride)
        for d in range(idx[0][0], idx[0][1]):
            for h in range(idx[1][0], idx[1][1]):
                for w in range(idx[2][0], idx[2][1]):
                    if self.verbose: print("  d, h, w:", d, h, w)
                    strided_idx = self._get_strided(d, h, w)
                    temp.append(self.data[strided_idx])

        if len(temp) > 1:
            return Matrix(idx[2][1] - idx[2][0], idx[0][1] - idx[0][0], idx[1][1] - idx[1][0], temp)
        else:
            return temp[0]

    def _get_strided(self, d, h, w):
        strided_idx = d * self.stride[0] + h * self.stride[1] + w * self.stride[2]
        if self.verbose: print("  strided:", strided_idx)
        return strided_idx
    
    def __setitem__(self, idx, item):
        d, h, w = idx
        strided_idx = self._get_strided(d, h, w)
        self.data[strided_idx] = item.data[0] if isinstance(item, Matrix) and item.size == 1 else item

    def __matmul__(self, other):
        assert self.shape[2] == other.shape[1], f"incompatible shapes for matmul: {self.shape[2]} != {other.shape[1]}" 
        assert self.shape[0] == other.shape[0], f"incompatible depth shapes for matmul: {self.shape[0]} != {other.shape[0]}"

        temp = Matrix(self.shape[0], self.shape[1], other.shape[2])
        if self.verbose: print(f"matmul on self={self.shape}, other={other.shape}")
        for d in range(self.shape[0]):
            for h in range(self.shape[1]):
                for w in range(other.shape[2]):
                    s = 0
                    for i in range(self.shape[2]):
                        if self.verbose: print(d, h, w, i)
                        s += self[d, h, i] * other[d, i, w]
                    
                    temp[d, h, w] = s

        return temp
    
    def _matrix_matrix_op(self, other, op):
        assert self.shape == other.shape, f"can only element-wise operate same sized matrices: {self.shape} != {other.shape}" 

        temp = Matrix(self.shape[0], self.shape[1], self.shape[2])
        for d in range(self.shape[0]):
            for h in range(self.shape[1]):
                for w in range(self.shape[2]):
                    temp[d, h, w] = op(self[d, h, w], other[d, h, w])

        return temp
    
    def __mul__(self, other):
        assert self.shape == other.shape, f"can only scalar multiply same sized matrices: {self.shape} != {other.shape}" 
        return self._matrix_matrix_op(other, lambda x, y: x * y)
    
    def __add__(self, other):
        assert self.shape == other.shape, f"can only scalar add same sized matrices: {self.shape} != {other.shape}" 
        return self._matrix_matrix_op(other, lambda x, y: x + y)
    
    def __sub__(self, other):
        assert self.shape == other.shape, f"can only scalar subtract same sized matrices: {self.shape} != {other.shape}" 
        return self._matrix_matrix_op(other, lambda x, y: x - y)
    
    def __truediv__(self, other):
        assert self.shape == other.shape, f"can only scalar divide same sized matrices: {self.shape} != {other.shape}" 
        return self._matrix_matrix_op(other, lambda x, y: x / y)
    
    def __floordiv__(self, other):
        assert self.shape == other.shape, f"can only scalar floor divide same sized matrices: {self.shape} != {other.shape}" 
        return self._matrix_matrix_op(other, lambda x, y: x // y)
    
    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return "Matrix("+str(self.__dict__)+")"

if __name__ == "__main__":
    m = Matrix(1, 2, 2, [i for i in range(1, 5)])
    print(m @ m)
    m2 = Matrix(1, 3, 3, [i for i in range(1, 10)], verbose=False)
    print(m2 @ m2)
    m3 = Matrix(1, 3, 2, [i for i in range(1, 7)], verbose=False)
    m4 = m2 @ m3
    print(m4[:, 1, :])
