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

---

## Exercise 2: PCA for Dimensionality Reduction in Classification

### Problem Overview
In this exercise, we explored the impact of **Principal Component Analysis (PCA)** on classification performance by reducing the dimensionality of high-dimensional data. Specifically, we aimed to understand whether dimensionality reduction can improve computational efficiency and maintain accuracy for classification tasks.

### Approach

1. **Dataset Preparation**: We used the MNIST dataset, which contains images of handwritten digits and is a high-dimensional dataset (784 features per sample). The dataset was split into training and testing sets.
2. **Dimensionality Reduction with PCA**: We applied PCA to the dataset to reduce the feature space while retaining 95% of the original variance. This allowed us to capture most of the data's information with fewer dimensions, improving efficiency.
3. **Classification Models**: We trained two classifiers:
   - **Logistic Regression**
   - **k-Nearest Neighbors (k-NN)** 
   on both the reduced-dimensional dataset and the original high-dimensional dataset.
4. **Performance Metrics**: We compared the classification accuracy and computation time for each model with and without PCA, to assess the impact of dimensionality reduction on both speed and accuracy.

### Results

| Model                   | Dataset             | Accuracy | Training Time (seconds) |
|-------------------------|---------------------|----------|--------------------------|
| Logistic Regression     | PCA-Reduced Data    | 0.9219   | 6.46                     |
| Logistic Regression     | Original Data       | 0.9159   | 7.33                     |
| k-Nearest Neighbors (k-NN) | PCA-Reduced Data | 0.9499   | 0.03                     |
| k-Nearest Neighbors (k-NN) | Original Data    | 0.9465   | 0.03                     |

### Analysis and Conclusions
- **Accuracy**: Both classifiers achieved comparable accuracy on the PCA-reduced data and the original data. Logistic Regression saw a slight accuracy improvement with PCA, indicating that dimensionality reduction removed some noise from the data.
- **Computation Time**: Training time decreased with PCA for Logistic Regression, demonstrating the computational benefits of reducing dimensionality. For k-NN, which is already efficient, PCA had a minimal effect on time.
- **Overall**: PCA proved to be beneficial for this dataset, as it reduced training time for Logistic Regression while maintaining high accuracy. This highlights PCA’s usefulness as a preprocessing step for high-dimensional data, especially for models sensitive to dimensionality.

---

## Exercise 3: k-Means Clustering vs. Gaussian Mixture Model (GMM)

### Problem Overview
In this exercise, we compared **k-Means Clustering** and the **Gaussian Mixture Model (GMM)** for unsupervised learning. Both methods were used to cluster a synthetic dataset with three distinct clusters. We evaluated the clustering quality based on inertia, log-likelihood, and the Silhouette Score.

### Approach

1. **Dataset Generation**: We generated a synthetic dataset with three clusters in a two-dimensional space, where each cluster had different standard deviations to simulate distinct shapes and densities.
2. **k-Means Clustering**: We applied k-means clustering with `k=3` clusters and evaluated the results using:
   - **Inertia**: Measures the compactness of clusters.
   - **Silhouette Score**: Reflects how well-separated the clusters are.
3. **Gaussian Mixture Model (GMM)**: We applied GMM with three components and evaluated the results using:
   - **Log-Likelihood**: Represents the likelihood of the data given the model, with higher values indicating a better fit.
   - **Silhouette Score**: For consistency, we used the same metric as for k-means to assess clustering quality.
4. **Visualization**: We visualized the clustering results for both methods. For GMM, we also plotted ellipses to represent the covariance of each Gaussian component, illustrating the flexibility of GMM in fitting clusters of various shapes.

### Results

| Method         | Evaluation Metric     | Score      |
|----------------|-----------------------|------------|
| k-Means        | Inertia               | 2444.12    |
| k-Means        | Silhouette Score      | 0.7737     |
| GMM            | Log-Likelihood        | -2016.59   |
| GMM            | Silhouette Score      | 0.7682     |

### Analysis and Conclusions
- **Silhouette Score**: k-Means achieved a slightly higher Silhouette Score than GMM, indicating a better clustering fit for this dataset with distinct, well-separated clusters.
- **Cluster Shapes**: The GMM plot shows ellipses representing each Gaussian component’s covariance, highlighting GMM's ability to model clusters with varying shapes and orientations. This flexibility can be advantageous for more complex datasets.
- **Overall**: For this dataset, **k-Means** provided a slightly better fit, as indicated by the higher Silhouette Score. However, **GMM** remains valuable for scenarios where clusters are less spherical or have different densities, as it can model more complex cluster shapes.
