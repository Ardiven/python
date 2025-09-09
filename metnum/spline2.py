import numpy as np
import matplotlib.pyplot as plt

# Titik kontrol
points = np.array([
    [0, 0],
    [1, 1],
    [2, -1],
    [3, 0],
    [4, -2],
    [5, 1]
])

# Knot vector
knots = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
degree = len(knots) - len(points) - 1  # = 2

# Basis function Cox–de Boor
def N(i, k, u):
    if k == 0:
        if (knots[i] <= u < knots[i+1]) or (u == knots[-1] and u == knots[i+1]):
            return 1.0
        return 0.0
    else:
        denom1 = knots[i+k] - knots[i]
        denom2 = knots[i+k+1] - knots[i+1]
        term1 = ((u - knots[i]) / denom1) * N(i, k-1, u) if denom1 != 0 else 0
        term2 = ((knots[i+k+1] - u) / denom2) * N(i+1, k-1, u) if denom2 != 0 else 0
        return term1 + term2

# Evaluasi titik pada parameter u
def bspline_point(u):
    n = len(points)
    result = np.zeros(2)
    for i in range(n):
        result += N(i, degree, u) * points[i]
    return result

# Buat kurva dengan sampling banyak u
us_dense = np.linspace(0, 1, 200)
curve = np.array([bspline_point(u) for u in us_dense])

# Titik evaluasi khusus di u=0.5
u_eval = 0.5
eval_point = bspline_point(u_eval)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(curve[:,0], curve[:,1], label='B-spline curve')
plt.plot(points[:,0], points[:,1], 'o--', color='grey',label='Control polygon')
plt.scatter(eval_point[0], eval_point[1], color='red', zorder=5, label=f'Point at u={u_eval}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic B-spline Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"Point at u = {u_eval}: {eval_point}")