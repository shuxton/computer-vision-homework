# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:26:28 2025

@author: Nikolai
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


# Zufällige 2D-Daten erzeugen
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 3]]  # Kovarianzmatrix
data = np.random.multivariate_normal(mean, cov, 500)

# Kovarianzmatrix berechnen
cov_matrix = np.cov(data.T)

# Eigenwerte und Eigenvektoren berechnen
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Plot erstellen
fig, ax = plt.subplots(figsize=(8, 8))

# Streudiagramm der Daten
ax.scatter(data[:, 0], data[:, 1], alpha=0.5, label="Datenpunkte", color='blue')

# Eigenvektoren plotten
for i in range(len(eigenvalues)):
    eigenvector = eigenvectors[:, i]  # Eigenvektor
    eigenvalue = eigenvalues[i]  # Eigenwert
    ax.quiver(0, 0, eigenvector[0], eigenvector[1], angles='xy', scale_units='xy', scale=1, 
              color='red', linewidth=2, label=f"Eigenvektor {i+1}")
    # Skalierter Eigenvektor entsprechend dem Eigenwert
    ax.quiver(0, 0, eigenvector[0] * eigenvalue, eigenvector[1] * eigenvalue, angles='xy', scale_units='xy', scale=1, 
              color='green', linewidth=2, alpha=0.7, label=f"Skalierter Eigenvektor {i+1}")

# Plot-Einstellungen
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal', adjustable='datalim')
ax.set_title("Daten mit Eigenvektoren und Kovarianzmatrix")
ax.legend(loc='upper left')
plt.grid()

plt.show()
#%%

# Erstellen eines einfachen Datensatzes mit zwei Clustern
X, _ = make_blobs(n_samples=100, centers=2, random_state=42)

# Plotten des ursprünglichen Datensatzes
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Datenpunkte')
plt.title('Originaler 2D-Datensatz')
plt.xlabel('Merkmal 1')
plt.ylabel('Merkmal 2')
plt.grid(True)
plt.show()

# PCA anwenden, um die Daten auf eine Hauptkomponente zu projizieren
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotten der transformierten Daten
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='red', label='PCA transformierte Daten')
plt.title('Daten nach PCA-Transformation')
plt.xlabel('Hauptkomponente 1')
plt.ylabel('Hauptkomponente 2')
plt.grid(True)
plt.show()

# Eigenwerte und Eigenvektoren anzeigen
print("Eigenwerte (Varianz der Hauptkomponenten):", pca.explained_variance_)
print("Eigenvektoren (Richtungen der Hauptkomponenten):", pca.components_)

# Visualisieren der Hauptkomponenten als Vektoren
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Originaldaten')
plt.quiver(0, 0, pca.components_[0, 0], pca.components_[0, 1], angles='xy', scale_units='xy', scale=1, color='red', label='Hauptkomponente 1')
plt.quiver(0, 0, pca.components_[1, 0], pca.components_[1, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Hauptkomponente 2')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('Hauptkomponenten als Vektoren')
plt.xlabel('Merkmal 1')
plt.ylabel('Merkmal 2')
plt.legend()
plt.grid(True)
plt.show()
#%%
%matplotlib inline
import matplotlib.pyplot as plt
from imageio.v3 import imread
from skimage import filters

# Load image
img = imread('images/pebbles.jpg')  # 'pebbles.jpg' or 'schrift.png'

# Apply Otsu's thresholding method
thresh_otsu = filters.threshold_otsu(img)
segments_otsu = img > thresh_otsu

# Apply Triangle method for thresholding
thresh_triangle = filters.threshold_triangle(img)
segments_triangle = img > thresh_triangle

# Plotting the results

plt.figure(figsize=(15, 15))

# Original image
plt.subplot(3, 2, 1)
plt.axis('off')
plt.imshow(img)
plt.title('Original Image')

# Histogram and Otsu's Threshold
plt.subplot(3, 2, 2)
plt.hist(img.flatten(), 256, (0, 255))
plt.axvline(thresh_otsu, color='r')
plt.title("Histogram with Otsu's Threshold")

# Thresholded image using Otsu's method
plt.subplot(3, 2, 3)
plt.axis('off')
plt.imshow(segments_otsu, cmap='gray')
plt.title("Segmented Image (Otsu's Threshold)")

# Histogram and Triangle Method Threshold
plt.subplot(3, 2, 4)
plt.hist(img.flatten(), 256, (0, 255))
plt.axvline(thresh_triangle, color='g')
plt.title("Histogram with Triangle Method Threshold")

# Thresholded image using Triangle method
plt.subplot(3, 2, 5)
plt.axis('off')
plt.imshow(segments_triangle, cmap='gray')
plt.title("Segmented Image (Triangle Threshold)")

plt.tight_layout()
plt.show()
