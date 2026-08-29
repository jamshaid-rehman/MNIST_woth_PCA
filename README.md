# MNIST PCA — Dimensionality Reduction & 3D Visualization

## 📌 Project Overview

This project was created mainly to understand **how Principal Component Analysis (PCA) works in practice**.

I used the **MNIST handwritten digit dataset** because it is a great example for understanding dimensionality reduction. Each MNIST image contains **784 pixel features**, which makes it a high-dimensional dataset.

The project explores how PCA transforms these 784 dimensions into a smaller number of meaningful components while preserving important information from the original data.

## 🎯 Main Objective

The main goal of this project was not simply to build a machine learning model, but to understand:

- How PCA works
- Why dimensionality reduction is useful
- How PCA transforms high-dimensional data
- How much information can be represented using fewer dimensions
- How high-dimensional data can be visualized in 3D

## 📊 Dataset

The project uses the **MNIST dataset**, which contains handwritten digits from **0 to 9**.

Each image is represented as:

- Image size: `28 × 28`
- Total features: `784`
- Classes: `0–9`

So, every image initially exists in a **784-dimensional feature space**.

## 🔬 PCA Implementation

The data is first standardized using `StandardScaler`.

Then PCA is applied to reduce the dimensionality:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_trf)
X_test_scaled = scaler.transform(X_test_trf)

pca = PCA(n_components=3)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
