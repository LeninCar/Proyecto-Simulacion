import numpy as np
from config import TOLERANCIA, MAX_ITERACIONES_LINEALES

def LU_decomposition(A):
    """Factorización LU de una matriz A"""
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
        L[i, i] = 1

    return L, U

def solve_LU(A, B):
    """Resuelve Ax = B usando factorización LU"""
    L, U = LU_decomposition(A)
    n = len(B)

    # Sustitución hacia adelante: L * Y = B
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])

    # Sustitución hacia atrás: U * X = Y
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - np.dot(U[i, i+1:], X[i+1:])) / U[i, i]

    return X

def Jacobi(A, b, max_iter=None):
    """Método de Jacobi para resolver Ax = b"""
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    
    n = b.shape[0]
    x = np.zeros(n)

    for k in range(max_iter):
        r = np.zeros(n)
        for i in range(n):
            v = 0
            for j in range(n):
                if j != i:
                    v += (A[i, j] * x[j]) / A[i, i]
            r[i] = b[i] / A[i, i] - v

        x = r.copy()

        if np.linalg.norm(A @ x - b) < TOLERANCIA:
            print(f"Jacobi convergió en {k+1} iteraciones")
            break

    return x

def GaussSeidel(A, b, x0=None, max_iter=None, tol=None):
    """Método de Gauss-Seidel para resolver Ax = b"""
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    if tol is None:
        tol = TOLERANCIA
    
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            suma = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Gauss-Seidel convergió en {k+1} iteraciones")
            return x_new

        x = x_new

    print("Gauss-Seidel no convergió dentro del número máximo de iteraciones")
    return x

def Richardson(A, b, max_iter=None):
    """Método de Richardson para resolver Ax = b"""
    if max_iter is None:
        max_iter = MAX_ITERACIONES_LINEALES
    
    n = b.shape[0]
    x = np.zeros(n)

    for k in range(max_iter):
        r = b - np.dot(A, x)
        x = x + r

        if np.linalg.norm(np.dot(A, x) - b) < TOLERANCIA:
            print(f"Richardson convergió en {k+1} iteraciones")
            break

    return x