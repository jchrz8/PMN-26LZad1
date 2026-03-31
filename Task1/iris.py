import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from scipy.stats import mode

# 1. Ładowanie danych
iris = datasets.load_iris()
X = iris.data  
y_true = iris.target # Prawdziwe etykiety z bazy (do metryk)

# 2. Inicjalizacja i dopasowanie modelu K-Means
kmeans = KMeans(n_clusters=3, random_state=1, n_init=10)
kmeans.fit(X)
# 3. Pobranie wyników
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

labels = np.zeros_like(y_kmeans)
for i in range(3):
    mask = (y_kmeans == i)
    labels[mask] = mode(y_true[mask], keepdims=True)[0]

print("--- METRYKI KLASYFIKACJI ---")
print(f"Accuracy (Dokładność): {accuracy_score(y_true, labels):.2f}")
print("\nPełny raport:")
print(classification_report(y_true, labels, target_names=iris.target_names))

tsne = TSNE(n_components=2, random_state=1, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Tworzymy dwa wykresy obok siebie dla porównania
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Wykres 1: Twoja wizualizacja cech płatka (Petal)
ax1.scatter(X[:, 2], X[:, 3], c=y_kmeans, s=50, cmap='viridis')
ax1.scatter(centers[:, 2], centers[:, 3], c='red', s=200, alpha=0.75, marker='X', label='Centroidy')
ax1.set_title("Grupowanie K-Means (Petal Length vs Width)")
ax1.set_xlabel(iris.feature_names[2])
ax1.set_ylabel(iris.feature_names[3])
ax1.legend()

# Wykres 2: Wizualizacja t-SNE (całość danych 4D rzutowana na 2D)
ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, s=50, cmap='viridis')
ax2.set_title("Wizualizacja t-SNE (Wszystkie 4 cechy)")
ax2.set_xlabel("Wymiar t-SNE 1")
ax2.set_ylabel("Wymiar t-SNE 2")

plt.tight_layout()
plt.show()