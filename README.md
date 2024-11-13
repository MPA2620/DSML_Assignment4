# Linear Regression: SVD vs. Gradient Descent Comparison

## Problem Overview
In this exercise, we compare two methods, **Singular Value Decomposition (SVD)** and **Gradient Descent (GD)**, to solve a linear regression problem. The goal is to understand and analyze the computational efficiency, convergence, and accuracy of each method in finding the optimal regression coefficients.

## Approach

### 1. Dataset Generation
To create a manageable and realistic dataset:
- **Dataset Size**: We generated 1000 samples with 10 features using scikit-learn's `make_regression` function. Gaussian noise was added to simulate real-world data.
- **Feature Matrix Adjustment**: Added a column of ones to account for the bias term in the linear regression model.
- **Model Representation**: Represented the regression model as \( y = X \beta + \epsilon \), where:
  - `X` is the feature matrix,
  - `y` is the target variable,
  - `β` is the vector of coefficients we aim to estimate.

### 2. Solving with SVD
SVD provides a direct, closed-form solution for linear regression by decomposing the matrix `X` into its singular values and vectors:
- **SVD Decomposition**: We decomposed `X` as \( X = U \Sigma V^T \).
- **Optimal Coefficients**: Used the pseudo-inverse formula, \( \beta = V \Sigma^{-1} U^T y \), to compute the regression coefficients efficiently.
- **Pros and Cons**: SVD is computationally efficient for small to medium datasets, giving an exact solution with minimal computation time.

### 3. Solving with Gradient Descent
Gradient Descent is an iterative optimization technique that finds the minimum of the Mean Squared Error (MSE) cost function:
- **Initialization**: Initialized `β` with zeros and set the learning rate and maximum iterations.
- **MSE Minimization**: Updated `β` iteratively to reduce the gradient of the cost function until it converges (or reaches a maximum iteration limit).
- **Stopping Criteria**: Used a gradient tolerance level to stop once updates became very small, indicating that we were near an optimal solution.
- **Trade-Off**: While Gradient Descent takes longer to converge compared to SVD, it remains useful in large datasets where SVD becomes less computationally feasible.

### 4. Comparison and Results
After obtaining solutions from both methods, we compared them based on:
- **Coefficients**: Both methods produced nearly identical coefficients, confirming their accuracy.
- **MSE**: Both solutions yielded similar Mean Squared Errors, validating the quality of the solutions.
- **Computation Time**: SVD was notably faster, as expected, due to its closed-form nature.
