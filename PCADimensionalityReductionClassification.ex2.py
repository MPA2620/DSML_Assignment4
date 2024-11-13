import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# Load dataset (e.g., MNIST for high-dimensional data)
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train classifiers on reduced-dimensional data (PCA)
print("Training classifiers on reduced dimensions using PCA:")

# Logistic Regression on PCA-reduced data
start_time = time.time()
lr_pca = LogisticRegression(max_iter=2000, random_state=42)
lr_pca.fit(X_train_pca, y_train)
lr_pca_time = time.time() - start_time
y_pred_lr_pca = lr_pca.predict(X_test_pca)
accuracy_lr_pca = accuracy_score(y_test, y_pred_lr_pca)

# k-Nearest Neighbors on PCA-reduced data
start_time = time.time()
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_pca.fit(X_train_pca, y_train)
knn_pca_time = time.time() - start_time
y_pred_knn_pca = knn_pca.predict(X_test_pca)
accuracy_knn_pca = accuracy_score(y_test, y_pred_knn_pca)

print(f"Logistic Regression on PCA-reduced data - Accuracy: {accuracy_lr_pca:.4f}, Time: {lr_pca_time:.4f} sec")
print(f"k-NN on PCA-reduced data - Accuracy: {accuracy_knn_pca:.4f}, Time: {knn_pca_time:.4f} sec")

# Train classifiers on original high-dimensional data (without PCA)
print("\nTraining classifiers on original high-dimensional data:")

# Logistic Regression on original data
start_time = time.time()
lr_original = LogisticRegression(max_iter=2000, random_state=42)
lr_original.fit(X_train_scaled, y_train)
lr_original_time = time.time() - start_time
y_pred_lr_original = lr_original.predict(X_test_scaled)
accuracy_lr_original = accuracy_score(y_test, y_pred_lr_original)

# k-Nearest Neighbors on original data
start_time = time.time()
knn_original = KNeighborsClassifier(n_neighbors=3)
knn_original.fit(X_train_scaled, y_train)
knn_original_time = time.time() - start_time
y_pred_knn_original = knn_original.predict(X_test_scaled)
accuracy_knn_original = accuracy_score(y_test, y_pred_knn_original)

print(f"Logistic Regression on original data - Accuracy: {accuracy_lr_original:.4f}, Time: {lr_original_time:.4f} sec")
print(f"k-NN on original data - Accuracy: {accuracy_knn_original:.4f}, Time: {knn_original_time:.4f} sec")

# Summary
print("\nSummary of Results:")
print("Reduced-Dimensional (PCA) Data:")
print(f"Logistic Regression - Accuracy: {accuracy_lr_pca:.4f}, Time: {lr_pca_time:.4f} sec")
print(f"k-Nearest Neighbors - Accuracy: {accuracy_knn_pca:.4f}, Time: {knn_pca_time:.4f} sec")

print("\nOriginal High-Dimensional Data:")
print(f"Logistic Regression - Accuracy: {accuracy_lr_original:.4f}, Time: {lr_original_time:.4f} sec")
print(f"k-Nearest Neighbors - Accuracy: {accuracy_knn_original:.4f}, Time: {knn_original_time:.4f} sec")
