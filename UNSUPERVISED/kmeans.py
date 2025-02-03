from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

open_file = "C:\\Users\\WAJI\\OneDrive\\projects\\PYTHON\\UNSUPERVISED\\kmeans_practice_dataset.csv"
df = pd.read_csv(open_file)
# Step 1: Extract features (drop the 'Cluster' column)
X = df[['Feature1', 'Feature2']]


inertia = []
cluster_range = range(1, 11)  # Trying for 1 to 10 clusters (you can adjust this range)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Step 3: Plot the Elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o', color='b', linestyle='--')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Step 2: Create a KMeans object and fit it to the data
kmeans = KMeans(n_clusters = 4, init= 'k-means++', n_init= 10)
X_means = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X['Feature1'], X['Feature2'], c=X_means, cmap='plasma', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title("K-means Predicted Clusters")
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.grid(True)
plt.show()
