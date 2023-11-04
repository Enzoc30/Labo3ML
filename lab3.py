import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from umap import umap_ as UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans

#Funciones y variables útiles incluidas en el repositorio del dataset

HEIGHT = 96
WIDTH = 96
DEPTH = 3
SIZE = HEIGHT * WIDTH * DEPTH
DATA_PATH = './data/stl10_binary/train_X.bin'
LABEL_PATH = './data/stl10_binary/train_y.bin'

def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images
    
def plot_image(image):
    plt.imshow(image)
    plt.show()

#PREGUNTA 1: Dataset

#re-escalamos la matriz pero afectaba demasiado la calidad de imagen
labels = read_labels(LABEL_PATH)
images = read_all_images(DATA_PATH)
categories = ["","airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"] #categorías en orden (1-10)

num_categories = 10
plt.figure(figsize=(20, 10))
for category in range(1, num_categories + 1):
    index = np.where(labels == category)[0][0] #encontramos el primero de cada categoria
    plt.subplot(1, num_categories, category)
    plt.imshow(images[index])
    plt.title(f"{category}: {categories[category]}")
    plt.axis('off')
plt.show()
plt.savefig('p1.png')

#------------------Obtenemos los datos X e Y--------------------------

path = "img/" #ruta Carpeta 

images = [] # Lista para almacenar las imágenes y las etiquetas
labels = []

for clse in range(1,11):
    ruta = os.path.join(path,str(clse))     #Obtener el path de la cada clase
    for archivo in os.listdir(ruta):     # Obtener los datos
        if(archivo.endswith(".png")):
            ruta_img = os.path.join(ruta,archivo)
            img = Image.open(ruta_img).convert("L")
            img_array = np.array(img)
            img_flat = img_array.flatten()
            images.append(img_flat)
            labels.append(clse)
            
X = np.array(images)
Y = np.array(labels)

#PREGUNTA 2: PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

categories = ["","airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"] # categorías en orden (1-10)
plt.figure(figsize=(12,12))
for i in np.unique(Y): #puntos con color basados en las clases
    plt.scatter(X_pca[Y == i, 0], X_pca[Y == i, 1], label=f'{categories[i]}')

eigenvectors = pca.components_.T # Plotear los eigenvectores
plt.quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], color='red', scale=3, label='Eigenvector 1')
plt.quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], color='blue', scale=3, label='Eigenvector 2')

plt.title('PCA')
plt.legend()
plt.show()
plt.savefig('p2.png')

#PREGUNTA 3: PCA + Embedding

umap = UMAP.UMAP(n_components=2)
X_umap = umap.fit_transform(X)
X_pcaE = pca.fit_transform(X_umap)

plt.figure(figsize=(12,12))
for i in np.unique(Y):
    plt.scatter(X_pcaE[Y == i, 0], X_pcaE[Y == i, 1], label=f'{categories[i]}')

plt.title('PCA + Embedding')
plt.legend()
plt.show()
plt.savefig('p3.png')

#PREGUNTA 4: t-SNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12,12))
for i in range(1, 11):
    mask = (Y == i)
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'{categories[i]}')

plt.title('t-SNE')
plt.legend()
plt.show()
plt.savefig('p4.png')

#PREGUNTA 5: t-SNE + Embedding

#FALTA

#PREGUNTA 6: K-means
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
y_kmeans_pca = kmeans.fit_predict(X_pca) #PCA
y_kmeans_tsne = kmeans.fit_predict(X_tsne) #t-SNE
y_kmeans_pcaE = kmeans.fit_predict(X_pcaE) #PCA + UMAP

cluster_labels = [str(i) for i in range(1, num_clusters + 1)]
plt.figure(12,12)

plt.subplot(2, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans_pca)
for i, label in enumerate(cluster_labels):
    plt.text(X_pca[y_kmeans_pca == i, 0].mean(), X_pca[y_kmeans_pca == i, 1].mean(), label,
             horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
plt.title('K-Means PCA')

plt.subplot(2, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans_tsne)
plt.title('K-Means t-SNE')
for i, label in enumerate(cluster_labels):
    plt.text(X_tsne[y_kmeans_tsne == i, 0].mean(), X_tsne[y_kmeans_tsne == i, 1].mean(), label,
             horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')

plt.subplot(2, 2, 3)
plt.scatter(X_pcaE[:, 0], X_pcaE[:, 1], c=y_kmeans_pcaE)
for i, label in enumerate(cluster_labels):
    plt.text(X_pcaE[y_kmeans_pcaE == i, 0].mean(), X_pcaE[y_kmeans_pcaE == i, 1].mean(), label,
             horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
plt.title('K-Means PCA + Embedding')

#FALTA

plt.tight_layout()
plt.show()
plt.savefig('p6.png')

#PREGUNTA 7: Meanshift

ms_pca = MeanShift(bandwidth=1200, bin_seeding=True)
y_ms_pca = ms_pca.fit_predict(X_pca) #PCA

ms_tsne = MeanShift(bandwidth=7.5, bin_seeding=True)
y_ms_tsne = ms_tsne.fit_predict(X_tsne) #t-SNE

ms_pcaE = MeanShift(bandwidth=0.95, bin_seeding=True)
y_ms_pcaE = ms_pcaE.fit_predict(X_pcaE) #PCA + UMAP

plt.figure(12,12)

plt.subplot(2, 2, 1)  # Use subplot position 2 for MeanShift clustering
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_ms_pca)
for i, label in enumerate(cluster_labels):
    plt.text(X_pca[y_ms_pca == i, 0].mean(), X_pca[y_ms_pca == i, 1].mean(), label,
             horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
plt.title('MeanShift PCA')

plt.subplot(2, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_ms_tsne)
for i, label in enumerate(cluster_labels):
    plt.text(X_tsne[y_ms_tsne == i, 0].mean(), X_tsne[y_ms_tsne == i, 1].mean(), label,
             horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
plt.title('MeanShift t-SNE')

plt.subplot(2, 2, 3)  # Use subplot position 2 for MeanShift clustering
plt.scatter(X_pcaE[:, 0], X_pcaE[:, 1], c=y_ms_pcaE)
for i, label in enumerate(cluster_labels):
    plt.text(X_pcaE[y_ms_pcaE == i, 0].mean(), X_pcaE[y_ms_pcaE == i, 1].mean(), label,
             horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
plt.title('MeanShift PCA + Embedding (UMAP)')

# FALTA

plt.tight_layout()
plt.show()
plt.savefig('p7.png')
