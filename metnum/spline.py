import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Control points
P = np.array([
    [0, 0],
    [1, 1],
    [2, -1],
    [3, 0],
    [4, -2],
    [5, 1]
])

# Knot vector
U = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], dtype=float)

# Degree
p = 2

# Create B-spline
spline = BSpline(U, P, p)

# Evaluate full curve
u_vals = np.linspace(U[p], U[-p-1], 200)
curve = spline(u_vals)

# Evaluate specific point
u_test = 0.5
point = spline(u_test)

# Plot
plt.plot(*curve.T, label="B-spline curve", color="blue")
plt.plot(*P.T, 'o--', label="Control polygon", color="gray")
plt.scatter(*point, color="red", label=f"Point at u={u_test}")
plt.title("Quadratic B-spline Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

print(f"Point at u = {u_test}: {point}")