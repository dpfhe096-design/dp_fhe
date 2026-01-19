import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

# Define sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Domain
# x_vals = np.linspace(-5, 5, 1000)
x_vals = np.linspace(-1, 1, 1000)
y_vals = sigmoid(x_vals)

# Fit Chebyshev polynomial
degree = 9
cheb = Chebyshev.fit(x_vals, y_vals, degree, domain=[-5, 5])
coeffs = cheb.convert().coef  # Convert to standard basis if needed

# Evaluate and plot
y_approx = cheb(x_vals)
max_error = np.max(np.abs(y_vals - y_approx))

# Print coefficients and error
print(f"Chebyshev coefficients (degree {degree}):")
for i, c in enumerate(coeffs):
    print(f"  c[{i}] = {c:.16f}")
# print(f"\nMax absolute error over [-5, 5]: {max_error:.6e}")
print(f"\nMax absolute error over [-1, 1]: {max_error:.6e}")

# Plot
plt.plot(x_vals, y_vals, label='sigmoid', linewidth=2)
plt.plot(x_vals, y_approx, '--', label=f'Chebyshev deg {degree}', linewidth=2)
plt.grid(True)
plt.legend()
plt.title('Chebyshev Approximation of Sigmoid')
plt.tight_layout()
plt.show()
