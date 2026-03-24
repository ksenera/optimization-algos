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

# Gaussian Elimination using Partial Pivoting solving Ax = b
def gaussElim(A, b):
    # declare A system and b vector 
    n = len(A)
    # want to modify only local copies of A sys and b vec 
    A = [row[:] for row in A]
    b = b[:]

    # from ALGO 2.4 for pivoting steps
    # for k = 1 to n - 1 {loop over cols.}
    for k in range(n-1):
        # find index p |Apk| >= |Aik| for k <= i <= n {search for pivot in current col}
        p = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[p][k]):
                p = i
        # if p != k then interchange rows k and p {interchange rows if necessary}
        if p != k:

            A[k], A[p] = A[p], A[k]
            b[k], b[p] = b[p], b[k]
            for row in A:
                print(row)
        # in Algo 2.4 this line is line 2 of Algo 2.3 
        # if akk = 0 then stop {stop if pivot is zero}
        if A[k][k] == 0:  # if a_kk = 0 then stop

            break

        # for j = k + 1 to n {compute multipliers for current col}
        for i in range(k+1, n):
            # mik = aik/akk 
            m_ik = A[i][k] / A[k][k]
            #print(f"Multiplier m_{i}{k} = {m_ik:.2f}")

            A[i][k] = 0 

            # for j = k + 1 to n 
            for j in range(k+1, n):
                # for i = k + 1 to n -> nested for j in for i
                # {apply transformation to remaining submatrix}
                A[i][j] = A[i][j] - m_ik * A[k][j] 

            # solving Ly = b (Step 2 page 12 by updating b)
            b[i] = b[i] - m_ik * b[k]

    x = backSub(A, b)
    return x

# from hints
def RayleighQuotient(A, x, tolerance):
    n = len(x)
    iterations = 0
    flag = True
    
    while flag:
        # sigma = (x^T A x) / (x^T x)
        Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        sigma = dot(x, Ax) / dot(x, x)
        # solve (A - sigma*I)y = x using gaussElim
        B = [row[:] for row in A]
        for i in range(n):
            B[i][i] -= sigma
        y = gaussElim(B, x[:])
        # xold = x
        xold = x
        # x = y / norm(y)
        x = [y[k] / norm(y) for k in range(n)]
        # if norm(x - xold) < tolerance: flag = False
        diff = [x[k] - xold[k] for k in range(n)]
        if norm(diff) < tolerance:
            return False
        # iterations += 1
        iterations += 1
    
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
        # extract column j from A
        v = [A[i][j] for i in range(n)]
        
        # subtract out projection of every already-locked vector
        for i in range(j):
            # measure pollution: dot(v, Q[i]) / dot(Q[i], Q[i])
            proj = dot(v, Q[i])/dot(Q[i], Q[i])
            # subtract it out
            v = [v[k] - proj * Q[i][k] for k in range(n)]
        
        # normalize: divide by magnitude
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