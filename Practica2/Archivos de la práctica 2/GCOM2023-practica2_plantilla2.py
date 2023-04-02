# -*- coding: utf-8 -*-
"""
Plantilla 2 de la práctica 2

Referencias:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    https://towardsdatascience.com/dbscan-d690cbae4c5d
    https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
"""

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

# #############################################################################
# Aquí tenemos definido el sistema X de 1500 elementos (personas) con dos estados
archivo1 = "C:/Users/rober/Downloads/Personas_en_la_facultad_matematicas.txt"
archivo2 = "C:/Users/rober/Downloads/Grados_en_la_facultad_matematicas.txt"
X = np.loadtxt(archivo1)
Y = np.loadtxt(archivo2)
labels_true = Y[:,0]

header = open(archivo1).readline()
print(header)
print(X)
#Si quisieramos estandarizar los valores del sistema, haríamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()
 
# #############################################################################
# Los clasificamos mediante el algoritmo DBSCAN
epsilon = 0.1

db = DBSCAN(eps=epsilon, min_samples=10, metric='manhattan').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


# #############################################################################
# Representamos el resultado con un plot

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
plt.show()