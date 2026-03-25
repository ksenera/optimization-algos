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
        print(k, x, f(x), g)
        if norm(g) < tol:
            break
        x = x + step_fn(H, g)
    return x


def newton_1d(f, df, ddf, x0, tol=1e-5):
    step = lambda H, g: -g / H          # scalar -f'/f''
    return newton(f, df, ddf, x0, step, tol)


def newton_nd(f, grad_f, hessian_f, x0, tol=1e-5):
    step = lambda H, g: solve_2x2(H, -g)   # vector solve H*s = -∇f
    return newton(f, grad_f, hessian_f, np.array(x0, dtype=float), step, tol)

if __name__ == "__main__":

    f1   = lambda x: x**2 - 2*x + 1
    df1  = lambda x: 2*x - 2
    ddf1 = lambda x: 2.0

    print(" Algorithm 1 (1-D) ")
    newton_1d(f1, df1, ddf1, x0=0.5)

    f2         = lambda x: 0.5*x[0]**2 + 2.5*x[1]**2
    grad_f2    = lambda x: np.array([x[0], 5.0*x[1]])
    hessian_f2 = lambda x: np.array([[1.0, 0.0],
                                      [0.0, 5.0]])

    print("\n Algorithm 2 (N-D) ")
    newton_nd(f2, grad_f2, hessian_f2, x0=[-1.0, 1.0])