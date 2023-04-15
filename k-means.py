from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=1000, centers=4, random_state=42)

# Perform k-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Print the centroids and cluster labels
print("Centroids:")
print(kmeans.cluster_centers_)
print("Labels:")
print(kmeans.labels_)
