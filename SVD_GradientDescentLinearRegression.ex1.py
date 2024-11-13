import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import time

# Generate dataset
n_features = 10
n_samples = 1000
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10, random_state=42)

# Add a column of ones to X for the bias term
X = np.c_[np.ones(n_samples), X]

# Solution using SVD
def solve_svd(X, y):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    Sigma_inv = np.diag(1 / Sigma)
    beta_svd = VT.T @ Sigma_inv @ U.T @ y
    return beta_svd

# Solution using Gradient Descent
def gradient_descent(X, y, learning_rate=0.01, max_iters=1000, tolerance=1e-6):
    n_samples, n_features = X.shape
    beta_gd = np.zeros(n_features)
    for iteration in range(max_iters):
        y_pred = X @ beta_gd
        gradient = -2 / n_samples * X.T @ (y - y_pred)
        beta_gd -= learning_rate * gradient

        # Check stopping criterion
        if np.linalg.norm(gradient) < tolerance:
            print(f"Converged in {iteration} iterations.")
            break
    return beta_gd

# Calculate solution and time for SVD
start_time = time.time()
beta_svd = solve_svd(X, y)
svd_time = time.time() - start_time

# Calculate solution and time for Gradient Descent
start_time = time.time()
beta_gd = gradient_descent(X, y, learning_rate=0.01, max_iters=10000, tolerance=1e-6)
gd_time = time.time() - start_time

# Calculate error (MSE) for both methods
y_pred_svd = X @ beta_svd
y_pred_gd = X @ beta_gd
mse_svd = mean_squared_error(y, y_pred_svd)
mse_gd = mean_squared_error(y, y_pred_gd)

# Display results
print("SVD Results:")
print(f"Coefficients beta: {beta_svd}")
print(f"MSE: {mse_svd}")
print(f"Computation time: {svd_time:.6f} seconds")

print("\nGradient Descent Results:")
print(f"Coefficients beta: {beta_gd}")
print(f"MSE: {mse_gd}")
print(f"Computation time: {gd_time:.6f} seconds")

# Comparison of obtained coefficients
print("\nCoefficient Comparison:")
print(f"Difference between SVD and GD coefficients: {np.linalg.norm(beta_svd - beta_gd)}")
