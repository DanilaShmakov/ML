from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
s = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>COLORS</title>
</head>
<body>
<table>
'''
i = 0

def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

data = pd.read_csv('colors.csv', delimiter=',')
print(data)
kmeans = KMeans(n_clusters=14)
y_pred = kmeans.fit_predict(data)
data = data.assign(label=y_pred)
print(data)
for item in kmeans.cluster_centers_:
    s += '<tr>'
    for index, col in data[y_pred == i].iterrows():
        print(col['r'], col['g'], col['b'])
        print(rgb_to_hex(col['r'], col['g'], col['b']))
        # s += f'''<td style="background-color:{col.mean}"></td>'''
        s += f'''<td style="background-color: {rgb_to_hex(col['r'], col['g'],col['b'])}">{index}</td>'''
    s += '<tr>'
    i += 1
s += '</table>'
with open('index.html', 'w') as file:
    file.write(s)


# plt.figure()
# plt.scatter(points[:, 0], points[:, 1], s=20, c=y_pred, cmap='rainbow')
# plt.title("Model data")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()
# print(kmeans.score(data))
#
# # kmeans = KMeans(n_clusters=5, n_init=10)
# # y_pred = kmeans.fit_predict(points)
# # plt.figure()
# # plt.scatter(points[:, 0], points[:, 1], s=20, c=y_pred, cmap='rainbow')
# # plt.title("Model data")
# # plt.xlabel("x1")
# # plt.ylabel("x2")
# # plt.show()
# # print(kmeans.score(points))
#
# from mlxtend.plotting import plot_decision_regions
# plt.subplots(figsize=(6, 6))
# plot_decision_regions(data, y_pred, clf=kmeans)
# plt.title("DATA")
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.legend(loc='upper left')
# plt.show()
#
# X_new = np.array([[0,0], [10,10], [0,10], [10,0]])
# kmeans.predict(X_new)
#
# kmeans.transform(X_new)


# points, labels = make_blobs(n_samples=1000, n_features=2,
# centers=[(20,20), (4,4)], cluster_std=2.0)

# print(kmeans.cluster_centers_)
# print(points.shape)
# print(points)
# print(labels)
