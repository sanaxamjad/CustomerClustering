# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates

# Load dataset
data = pd.read_csv('mall_customers.csv')

# Preview the dataset
print(data.head())

# Data Cleaning: Convert categorical data
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Standardize the data to normalize the scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):  # Test k values from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid()
plt.show()

# Based on the elbow curve, select the optimal k (e.g., k=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# Visualize the clusters using Spending Score and Annual Income
plt.figure(figsize=(10, 7))
sns.scatterplot(x=data['Annual Income (k$)'], 
                y=data['Spending Score (1-100)'], 
                hue=data['Cluster'], 
                palette='viridis', 
                s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Cluster Distribution by Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', hue='Gender', data=data, palette='viridis')
plt.title('Cluster Distribution by Gender')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.grid()
plt.show()

# Boxplots for Each Feature by Cluster
# Age Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Age', data=data, palette='viridis')
plt.title('Age Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Age')
plt.grid()
plt.show()

# Annual Income Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Annual Income (k$)', data=data, palette='viridis')
plt.title('Annual Income Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Annual Income (k$)')
plt.grid()
plt.show()

# Spending Score Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=data, palette='viridis')
plt.title('Spending Score Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Spending Score (1-100)')
plt.grid()
plt.show()

# Cluster Size Distribution (Cluster Countplot)
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=data, palette='viridis')
plt.title('Cluster Size Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.grid()
plt.show()

# Histograms for Each Feature by Cluster
# Age Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Cluster', multiple='stack', palette='viridis', bins=15)
plt.title('Age Distribution by Cluster')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid()
plt.show()

# Annual Income Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Annual Income (k$)', hue='Cluster', multiple='stack', palette='viridis', bins=15)
plt.title('Annual Income Distribution by Cluster')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Count')
plt.grid()
plt.show()

# Spending Score Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Spending Score (1-100)', hue='Cluster', multiple='stack', palette='viridis', bins=15)
plt.title('Spending Score Distribution by Cluster')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Count')
plt.grid()
plt.show()

# Silhouette Score Plot
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f'Silhouette Score: {silhouette_avg}')

# Dendrogram for Hierarchical Clustering (Optional for comparison)
Z = linkage(X_scaled, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# 3D Scatter Plot for Clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis', s=100)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D Scatter Plot of Clusters')
plt.show()

# Pair Plot for Features
sns.pairplot(data, hue='Cluster', palette='viridis', vars=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
plt.title('Pair Plot of Features by Cluster')
plt.show()

# Centroids Visualization
centroids = kmeans.cluster_centers_
plt.figure(figsize=(10, 7))
sns.scatterplot(x=data['Annual Income (k$)'], 
                y=data['Spending Score (1-100)'], 
                hue=data['Cluster'], 
                palette='viridis', 
                s=100)
plt.scatter(centroids[:, 1], centroids[:, 2], c='red', s=200, marker='X', label='Centroids')
plt.title('Customer Segments with Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Correlation Heatmap
corr = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title('Correlation Heatmap of Features')
plt.show()

# Parallel Coordinates Plot for Cluster Separation
plt.figure(figsize=(10, 6))
parallel_coordinates(data, 'Cluster', cols=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], color=plt.cm.viridis.colors)
plt.title('Parallel Coordinates Plot of Clusters')
plt.xlabel('Features')
plt.ylabel('Values')
plt.grid()
plt.show()

# Save the clustered dataset
data.to_csv('mall_customers_with_clusters.csv', index=False)