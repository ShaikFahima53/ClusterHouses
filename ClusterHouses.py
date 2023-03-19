# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt")

# Data cleaning & getting rid of irrelevant information before clustering
df.drop(['Floor', 'TotalFloor', 'Bathroom', 'Living.Room'], axis=1, inplace=True)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Finding the optimal value of k
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(12, 8))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs Number of Clusters')
plt.show()

# Train the model with the best value of k
k = 4  # based on the elbow plot
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)

# Storing cluster to which the house belongs along with the data
df['Cluster'] = kmeans.labels_
print(df.head())
