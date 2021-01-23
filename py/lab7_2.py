import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.impute import SimpleImputer

data = pd.read_csv('IoT.csv', delimiter=',')
quantity = ['ts','duration', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp',
          'missed_bytes', 'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
quality = ['uid', 'id.orig_h',  'id.orig_p', 'id.resp_h', 'id.resp_p',
         'proto','service', 'conn_state', 'history', 'tunnel_parents']
# print(data)

for i in quantity:
    k = 0
    for item in data[i]:
        if item == '-':
            data[i][k] = np.nan
        data[i][k] = float(data[i][k])
        k += 1

# только эти 3 признака содержат '-'
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data[['duration', 'orig_bytes', 'resp_bytes']] = imp.fit_transform(data[['duration', 'orig_bytes', 'resp_bytes']])

# так как есть полностью пустые признаки меняем их на нули
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
data[data.columns] = imp.fit_transform(data[data.columns])


le = LabelEncoder()
for item in data[quality]:
    data[item] = le.fit_transform(data[item])


ss = StandardScaler()
data.iloc[:, :-2] = ss.fit_transform(data.iloc[:, :-2])


pca = PCA(svd_solver='full')
data2 = data.copy()

сolumn = []
for i in range(len(data.columns) -2):
    сolumn.append(f'pc{i+1}')
сolumn.append('label' )
сolumn.append('detailed-label')
data2.columns = сolumn
data2.iloc[:, :-2] = pca.fit_transform(data2.iloc[:, :-2])

# print(data2)

plt.figure()
plt.grid()
plt.scatter(data2['pc1'], data2['pc2'], c=le.fit_transform(data2['label']), lw=.6, edgecolors='black')
plt.axis('equal')
plt.title("Данные в пространстве первых двух главных компонент")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()

plt.matshow(pca.components_[:2])
plt.colorbar()
plt.gca().xaxis.tick_bottom()
plt.xticks(range(len(data.columns) - 2), data.iloc[:, :-2], rotation=90)
plt.yticks(range(2), data2[['pc1', 'pc2']])
plt.title("Тепловая карта")
plt.show()