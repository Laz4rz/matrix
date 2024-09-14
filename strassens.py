from matrix import Matrix

def split(m):
    n = m.shape[1] // 2
    a = m[:, :n, :n]
    b = m[:, n:, :n]
    c = m[:, :n, n:]
    d = m[:, n:, n:]
    return a, b, c, d

def strassens_dnq(A, B):
    """
    Vanilla divide and conquer approach, with time complexity same
    as normal matmul, for testing purposes
    """
    if not A.shape == B.shape and A.shape[1]//2 == 0:
        return A @ B
    elif A.shape[1] <= 2:
        return A @ B
    else:
        a, b, c, d = split(A)
        e, f, g, h = split(B)

        ae = strassens_dnq(a, e)
        bg = strassens_dnq(b, g)
        af = strassens_dnq(a, f)
        bh = strassens_dnq(b, h)
        ce = strassens_dnq(c, e)
        dg = strassens_dnq(d, g)
        cf = strassens_dnq(c, f)
        dh = strassens_dnq(d, h)

        C11 = ae + bg
        C12 = af + bh
        C21 = ce + dg
        C22 = cf + dh

        C = Matrix(C11.shape[0], C11.shape[1], C11.shape[1], 
                data=C11.data + C12.data + C21.data + C22.data           
            )

        return C

def strassens(A, B):
    if not A.shape != B.shape and A.shape[1]//2 != 0:
        return A @ B
    elif A.shape[1] <= 2:
        return A @ B
    else:
        a, b, c, d = split(A)
        e, f, g, h = split(B)

        p1 = strassens(a, f - h)
        p2 = strassens(a + b, h)
        p3 = strassens(c + d, e)
        p4 = strassens(d, g - e)
        p5 = strassens(a + d, e + h)
        p6 = strassens(b - d, g + h)
        p7 = strassens(a - c, e + f)

        C11 = p5 + p4 - p2 + p6
        C12 = p1 + p2
        C21 = p3 + p4
        C22 = p1 + p5 - p3 - p7

        C = Matrix(C11.shape[0], C11.shape[1], C11.shape[1], 
                data=C11.data + C12.data + C21.data + C22.data           
            )

        return C
