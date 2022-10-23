import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file = "/content/mall_customer.csv"

df = pd.read_csv(file)
df.head()

features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]

plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score']);

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

X['Annual_Income_(k$)']

X

X['Spending_Score']

#plt.scatter(X[0], X[1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
centers
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
