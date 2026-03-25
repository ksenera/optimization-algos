import numpy as np

def solve_2x2(H, g):
    a, b = H[0,0], H[0,1]
    c, d = H[1,0], H[1,1]
    det  = a*d - b*c
    return np.array([(d*g[0] - b*g[1]) / det,
                     (a*g[1] - c*g[0]) / det])

def norm(v):
    if np.isscalar(v):
        return abs(v)
    return (v[0]**2 + v[1]**2) ** 0.5

def compute_step(H, g):
    if np.isscalar(H):
        return -g / H
    else:
        return solve_2x2(H, -g)

# NEWTON reusable from ALGO 6.1 
#x₀ = initial guess
#for k = 0, 1, 2, ...
#    x_{k+1} = x_k - f'(x_k) / f''(x_k)
#end
def newton(f, grad_f, hessian_f, x0, step_fn, tol=1e-5, max_iter=1000):
    x = x0
    for k in range(max_iter):
        g = grad_f(x)
        H = hessian_f(x)
        if np.isscalar(x):
            print(f"{k:<5}{x:<15.6f}{f(x):<15.6f}{g:<15.6f}")
        else:
            xy  = f"({x[0]:.4f}, {x[1]:.4f})"
            gxy = f"[{g[0]:.4f}, {g[1]:.4f}]"
            print(f"{k:<5}{xy:<20}{f(x):<15.6f}{gxy:<25}")
        if norm(g) < tol:
            break
        s = compute_step(H, g)
        x = x + s
    return x


def newton_1d(f, df, ddf, x0, tol=1e-5):
    header = "f'(x)"
    print(f"{'k':<5}{'x':<15}{'f(x)':<15}{header:<15}")
    print("-" * 50)
    return newton(f, df, ddf, x0, tol)


def newton_nd(f, grad_f, hessian_f, x0, tol=1e-5):
    print(f"{'k':<5}{'(x, y)':<20}{'f(x,y)':<15}{'grad f(x,y)':<25}")
    print("-" * 65)
    return newton(f, grad_f, hessian_f, np.array(x0, dtype=float), tol)


def f1(x):
    return x**2 - 2*x + 1
 
def derivative_f1(x):
    return 2*x - 2
 
def double_deri_f1(x):
    return 2.0

if __name__ == "__main__":

    print(" Algorithm 1 (1-D) ")
    newton_1d(f1, df1, ddf1, x0=0.5)

    print("\n Algorithm 2 (N-D) ")
    newton_nd(f2, grad_f2, hessian_f2, x0=[-1.0, 1.0])