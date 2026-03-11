"""
PROJECT   : A4.py
PROGRAMMER: Kushika Senera
COURSE    : SFWRTECH 4MA3 - Numerical Linear Algebra and Numerical Optimization
INSTRUCTOR: Gagan Sidhu
DATE: Tuesday, March 10th 
"""

import numpy as np

# lower triangle system Lx = b
def forwardSub(L, b):
    # declare and initialize output vector x
    n = len(b)
    x = [0.0] * n
    # for j = 1 to n {loop over cols.} 
    # got length of b vector and put it in n to start loop
    for j in range(n):
        # xj = bj/Ljj {compute soln. component}
        x[j] = b[j] / L[j][j]

        # for i = j + 1 to n 
        # j starts at index 0 so j + 1 = 1 & n = 4 so from index 1 up to index 3 
        # range start to stop index n-1 => 4-1 = 3 
        for i in range(j+1,n):
            # bi = bi - Lijxj {update RHS}
            b[i] = b[i] - L[i][j] * x[j]
    return x

# upper triangle system Ux = b 
def backSub(U, b):
    # declare and initialize output vector x
    n = len(b)
    x = [0.0] * n
    # for j = n to 1 {loop backwards over cols.} 
    for j in reversed(range(n)):
        # xj = bj/Ujj {compute soln. component}
        x[j] = b[j] / U[j][j]
        # for i = 1 to j - 1
        # range starts at 1 and stops at j-1
        # using range(stop) => j => j-1 
        for i in range(j):
            # bi = bi - uijxj {update RHS}
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
    pass

def qrIteration(A, tolerance):
    pass

def gramSchmidt(A):
    n = len(A)
    Q = []
    
    for j in range(n):
        # extract column j from A
        v = Q.append(A[i][j])
        
        # subtract out projection of every already-locked vector
        for i in range(j):
            # measure pollution: dot(v, Q[i]) / dot(Q[i], Q[i])
            proj = np.dot(v, Q[i])/np.dot(Q[i], Q[i])
            # subtract it out
            v -= proj
        
        # normalize: divide by magnitude
        mag = 1
        q = [v[k] / mag for k in range(n)]
        Q.append(q)
    
    return Q

def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))