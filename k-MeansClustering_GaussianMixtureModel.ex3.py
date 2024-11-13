import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Generate synthetic dataset with three clusters
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)

# k-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
kmeans_inertia = kmeans.inertia_
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# Plot k-means clustering results
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=30, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("k-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)
gmm_log_likelihood = gmm.score(X) * X.shape[0]  # Total log-likelihood
gmm_silhouette = silhouette_score(X, gmm_labels)

# Plot GMM clustering results with ellipses
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, s=30, cmap='viridis')
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot GMM covariance ellipses
def plot_ellipse(position, covariance, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if covariance.shape == (2, 2):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = np.arctan2(*eigenvectors[0][::-1])
        width, height = 2 * np.sqrt(eigenvalues)
        for nSig in range(1, 4):
            ax.add_patch(plt.matplotlib.patches.Ellipse(
                position, nSig * width, nSig * height, angle=angle * 180 / np.pi, **kwargs
            ))

for i in range(3):
    plot_ellipse(gmm.means_[i], gmm.covariances_[i], alpha=0.3, ax=plt.gca(), color='red')

plt.tight_layout()
plt.show()

# Display results
print("k-Means Clustering Results:")
print(f"Inertia: {kmeans_inertia}")
print(f"Silhouette Score: {kmeans_silhouette:.4f}")

print("\nGaussian Mixture Model Results:")
print(f"Log-Likelihood: {gmm_log_likelihood:.4f}")
print(f"Silhouette Score: {gmm_silhouette:.4f}")

# Summary and Analysis
if gmm_silhouette > kmeans_silhouette:
    print("\nGMM provided a better clustering fit based on the Silhouette Score.")
else:
    print("\nk-Means provided a better clustering fit based on the Silhouette Score.")
