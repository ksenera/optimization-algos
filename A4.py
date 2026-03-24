"""
PROJECT   : A4.py
PROGRAMMER: Kushika Senera
COURSE    : SFWRTECH 4MA3 - Numerical Linear Algebra and Numerical Optimization
INSTRUCTOR: Gagan Sidhu
DATE: Tuesday, March 10th 
"""

A_matrix = [[2.9766, 0.3945, 0.4198, 1.1159],
            [0.3945, 2.7328, -0.3097, 0.1129],
            [0.4198, -0.3097, 2.5675, 0.6079],
            [1.1159, 0.1129, 0.6079, 1.7231]]


def forwardSub(L, b):
    n = len(b)
    x = [0.0] * n
    for j in range(n):
        x[j] = b[j] / L[j][j]
        for i in range(j+1,n):
            b[i] = b[i] - L[i][j] * x[j]
    return x


def backSub(U, b):
    n = len(b)
    x = [0.0] * n
    for j in reversed(range(n)):
        if abs(U[j][j]) < 1e-12:
            x[j] = 0.0
        else:
            x[j] = b[j] / U[j][j]
        for i in range(j):
            b[i] = b[i] - U[i][j] * x[j]
    return x

def gaussElim(A, b):
    n = len(A)
    A = [row[:] for row in A]
    b = b[:]
    for k in range(n-1):
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i
        if p != k:
            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
        if abs(A[k][k]) < 1e-12: 
            break
        for i in range(k+1, n):
            m_ik = A[i][k] / A[k][k]
            A[i][k] = 0 
            for j in range(k+1, n):
                A[i][j] = A[i][j] - m_ik * A[k][j] 
            b[i] = b[i] - m_ik * b[k]
    return backSub(A, b)

# from hints
def RayleighQuotient(A, x, tolerance):
    n = len(x)
    iterations = 0
    flag = True
    while flag:
        Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        sigma = dot(x, Ax) / dot(x, x)
        B = [row[:] for row in A]
        for i in range(n):
            B[i][i] -= sigma
        y = gaussElim(B, x[:])
        mag = norm(y)
        if mag < 1e-12:
            flag = False
        else:
            xold = x[:]
            x = [y[k] / mag for k in range(n)]
            iterations += 1
            diff1 = norm([x[k] - xold[k] for k in range(n)])
            diff2 = norm([x[k] + xold[k] for k in range(n)])
            if min(diff1, diff2) < tolerance:
                flag = False
    return sigma, iterations
    

def qrIteration(A, tolerance):
    n = len(A)
    eigenvalues = [1.0] * n
    iterations = 0
    flag = True
    while flag:
        # Use QR = A to determine Q and R via Householder transformation
        Q = gramSchmidt(A)
        R = [[sum(Q[i][k] * A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        # Anew = R*Q
        Anew = [[sum(R[i][k] * Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        # eignValusNew = diagonal elements of Anew
        eigenvalNew = [Anew[i][i] for i in range(n)]
        iterations += 1 
        # if norm(eigenvalues – eignValusNew)<tolerance:
        diff = norm([eigenvalNew[i] - eigenvalues[i] for i in range(n)])
        if diff < tolerance:
            return eigenvalNew, iterations
        A = Anew
        eigenvalues = eigenvalNew
    return eigenvalues, iterations

def gramSchmidt(A):
    n = len(A)
    Q = []
    for j in range(n):
        v = [A[i][j] for i in range(n)]
        for i in range(j):
            proj = dot(v, Q[i])/dot(Q[i], Q[i])
            v = [v[k] - proj * Q[i][k] for k in range(n)]
        mag = dot(v, v) ** 0.5
        q = [v[k] / mag for k in range(n)]
        Q.append(q)
    return Q

def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def norm(x):
    return sum(x[i]**2 for i in range(len(x))) ** 0.5


if __name__ == "__main__":
    tol = 0.0001
    vectors = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    names = ["e1","e2","e3","e4"]

    print("# Starting Eigen vector Eigen value Number of iterations")
    for idx in range(4):
        sigma, iters = RayleighQuotient(A_matrix, vectors[idx], tol)
        print(idx+1, names[idx], sigma, iters)

    eigenvalues, iterations = qrIteration(A_matrix, tol)
    print("The eigen values are:", eigenvalues)
    print("The number of iterations for the convergence is:", iterations)