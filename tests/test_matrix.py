import unittest

from matrix import Matrix


class TestMatrix(unittest.TestCase):

    def test_matrix_multiplication(self):
        # vector x vector
        with self.subTest("vector x vector"):
            m1 = Matrix(1, 1, 5, [1, 2, 3, 4, 5])
            m2 = Matrix(1, 5, 1, [1, 2, 3, 4, 5])
            result = m1 @ m2
            self.assertEqual(result.data, [55])
            self.assertEqual(result.shape, (1, 1, 1))

        # matrix x vector
        with self.subTest("matrix x vector"):
            m3 = Matrix(1, 3, 3, [i for i in range(9)])
            m4 = Matrix(1, 3, 1, [1, 2, 3])
            result = m3 @ m4
            self.assertEqual(result.data, [8, 26, 44])
            self.assertEqual(result.shape, (1, 3, 1))

        # matrix x matrix (2D square)
        with self.subTest("matrix x matrix (2D square)"):
            m5 = Matrix(1, 3, 3, [i for i in range(9)])
            m6 = Matrix(1, 3, 3, [i for i in range(9)])
            result = m5 @ m6
            self.assertEqual(result.data, [15, 18, 21, 42, 54, 66, 69, 90, 111])
            self.assertEqual(result.shape, (1, 3, 3))

        # matrix x matrix (2D non-square)
        with self.subTest("matrix x matrix (2D non-square)"):
            m7 = Matrix(1, 3, 2, [i for i in range(6)])
            m8 = Matrix(1, 2, 3, [i for i in range(6)])
            result = m7 @ m8
            # just to be sure, can be used whenever you want to double check
            import numpy as np
            m = np.array([i for i in range(6)]).reshape(3, 2)
            n = np.array([i for i in range(6)]).reshape(2, 3)
            o = m @ n
            self.assertEqual(result.data, list(o.flatten()))
            self.assertEqual(result.data, [3, 4, 5, 9, 14, 19, 15, 24, 33])
            self.assertEqual(result.shape, (1, 3, 3))

        # matrix x matrix (3D square)
        with self.subTest("matrix x matrix (3D square)"):
            m9 = Matrix(2, 2, 2, [i for i in range(8)])
            m10 = Matrix(2, 2, 2, [i for i in range(8)])
            result = m9 @ m10
            self.assertEqual(result.data, [2, 3, 6, 11, 46, 55, 66, 79])
            self.assertEqual(result.shape, (2, 2, 2))

        # matrix x matrix (3D non-square)
        with self.subTest("matrix x matrix (3D non-square"):
            m11 = Matrix(2, 3, 2, [i for i in range(12)])
            m12 = Matrix(2, 2, 3, [i for i in range(12)])
            result = m11 @ m12
            self.assertEqual(result.data, [3, 4, 5, 9, 14, 19, 15, 24, 33, 99, 112, 125, 129, 146, 163, 159, 180, 201])
            self.assertEqual(result.shape, (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
