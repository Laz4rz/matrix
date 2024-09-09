import torch


class Matrix:
    # 3d implemented with strided representation in row-major order
    def __init__(self, h=1, w=1, d=1, data=None):
        self.shape = (h, w, d)
        self.stride = (h * d, w, 1)
        if data is not None:
            assert len(data) == h * w * d
        else:
            self.data = [0 for _ in range(h) for _ in range(d) for _ in range(w)]
            self.data = [1 ,2 ,3 ,4]


    def __getitem__(self, idx):
        idx = list(idx) + [0 for _ in range(3-len(idx))]

        for i in range(len(idx)):
            if isinstance(idx[i], slice):
                start, stop = idx[i].start, idx[i].stop
                start = 0 if start is None else start
                stop = self.shape[i] if stop is None else stop
                idx[i] = (start, stop)
            else:
                idx[i] = (idx[i], idx[i] + 1)
            print(idx)
            
        temp = []
        for h in range(idx[0][0], idx[0][1]):
            for w in range(idx[1][0], idx[1][1]):
                for d in range(idx[2][0], idx[2][1]):
                    print("h, w, d:", h, w, d)
                    print(self.stride)
                    strided_idx = d * self.stride[0] + h * self.stride[1] + w * self.stride[2]
                    print("strided:",strided_idx)
                    temp.append(self.data[strided_idx])

        return temp

    
    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return "Matrix("+str(self.__dict__)+")"
    
m = Matrix(2, 2)
print(m)
m[:, :]
