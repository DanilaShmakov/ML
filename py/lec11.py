import pandas as pd
#import numpy as np

df = pd.read_csv('moons.csv')
X0 = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(X0)
X = ss.transform(X0)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, init='k-means++')
y_pred = kmeans.fit_predict(X)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df['x1'], df['x2'], c=y_pred, s=20, cmap='rainbow')
plt.title("Clusters")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.savefig('pic1.pdf')
plt.show()

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=2)
y_pred = agg.fit_predict(X)

#import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df['x1'], df['x2'], c=y_pred, s=20, cmap='rainbow')
plt.title("Clusters")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.savefig('pic2.pdf')
plt.show()

from scipy.cluster.hierarchy import dendrogram, ward

linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
ax.get_xaxis().set_visible(False)
bounds = ax.get_xbound()
plt.title("Dendrogram")
plt.savefig('pic3.pdf')
plt.show()

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.3, min_samples=10)
y_pred = dbscan.fit_predict(X) # шум: -1
print(y_pred)

plt.figure()
plt.scatter(df['x1'], df['x2'], c=y_pred, s=20, cmap='rainbow')
plt.title("Clusters")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis('equal')
plt.savefig('pic4.pdf')
plt.show()

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_title(people.target_names[target])
plt.suptitle("some_faces")
plt.savefig('_pic1.pdf')
plt.show()


print(people.images.shape)
print(format(len(people.target_names)))

import numpy as np

counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end=' ')
    if (i + 1) % 3 == 0:
        print()

"""	
"""

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:20]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.

from sklearn.decomposition import PCA

pca = PCA(n_components=100, whiten=True, random_state=0)

pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=15, min_samples=3)
labels = dbscan.fit_predict(X_pca)

print("\nlabels: {}".format(np.unique(labels)))
print("capacity: {}".format(np.bincount(labels+ 1)))

# outlier detection
noise = X_people[labels==-1]
fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()},figsize=(12, 4))
for image, ax in zip(noise, axes.ravel()):
    ax.imshow(image.reshape(image_shape), vmin=0, vmax=1, cmap=plt.cm.gray)
plt.savefig('_pic2.pdf')
plt.show()

for eps in [1, 3, 5, 7, 9, 11, 13]:
    print("\neps={}".format(eps))
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print("labels: {}".format(np.unique(labels)))
    print("capacity: {}".format(np.bincount(labels + 1)))


dbscan = DBSCAN(eps=8, min_samples=3)
labels = dbscan.fit_predict(X_pca)
for cluster in range(max(labels) + 1):
    mask = labels == cluster
    n_images = np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks': (), 'yticks': ()})
    for image, label, ax in zip(X_people[mask], y_people[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1, cmap=plt.cm.gray)
        ax.set_title(people.target_names[label].split()[-1])
plt.savefig('_pic3.pdf')
plt.show()

from sklearn.cluster import KMeans

km = KMeans(n_clusters=10, random_state=0)
labels = km.fit_predict(X_pca)
print("\ncapacity: {}".format(np.bincount(labels)))

fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape),vmin=0, vmax=1, cmap=plt.cm.gray)
plt.savefig('_pic4.pdf')
plt.show()

def plot_kmeans_faces(km, pca, X_pca, X_people, y_people, target_names):
    n_clusters = 10
    image_shape = (87, 65)
    fig, axes = plt.subplots(n_clusters, 11, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(10, 15), gridspec_kw={"hspace": .3})

    for cluster in range(n_clusters):
        center = km.cluster_centers_[cluster]
        mask = km.labels_ == cluster
        dists = np.sum((X_pca - center) ** 2, axis=1)
        dists[~mask] = np.inf
        inds = np.argsort(dists)[:5]
        dists[~mask] = -np.inf
        inds = np.r_[inds, np.argsort(dists)[-5:]]
        axes[cluster, 0].imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1, cmap=plt.cm.gray)
        for image, label, asdf, ax in zip(X_people[inds], y_people[inds], km.labels_[inds], axes[cluster, 1:]):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1, cmap=plt.cm.gray)
            ax.set_title("%s" % (target_names[label].split()[-1]), fontdict={'fontsize': 9})

    # add some boxes to illustrate which are similar and which dissimilar
    rec = plt.Rectangle([-5, -30], 73, 1295, fill=False, lw=2)
    rec = axes[0, 0].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 0].text(0, -40, "Center")

    rec = plt.Rectangle([-5, -30], 385, 1295, fill=False, lw=2)
    rec = axes[0, 1].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 1].text(0, -40, "Close to center")

    rec = plt.Rectangle([-5, -30], 385, 1295, fill=False, lw=2)
    rec = axes[0, 6].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 6].text(0, -40, "Far from center")

    plt.savefig('_pic5.pdf')
    plt.show()


plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=40) #10
labels = agg.fit_predict(X_pca)
print("\ncapacity: {}".format(np.bincount(labels)))

n_clusters= 40
#for cluster in range(n_clusters):
for cluster in[10, 13, 19, 22, 36]:
    mask = labels == cluster
    fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
    cluster_size = np.sum(mask)
    axes[0].set_ylabel(cluster_size)
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1, cmap=plt.cm.gray)
        ax.set_title("%s" % (people.target_names[label].split()[-1]), fontdict={'fontsize': 9})
        for i in range(cluster_size, 15):
            axes[i].set_visible(False)

plt.savefig('_pic6.pdf')
plt.show()
