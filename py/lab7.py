import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import numpy as np
import funcs

data = pd.read_csv('IoT.csv', delimiter=',')
quantity = ['ts','duration','orig_bytes', 'resp_bytes','local_orig','local_resp',
     'missed_bytes','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes']
quality = ['uid','id.orig_h',  'id.orig_p', 'id.resp_h', 'id.resp_p','proto','service',
      'conn_state','history', 'tunnel_parents']

print(data)
# data = pd.DataFrame(data, columns=quantity+quality)

for i in data[quantity]:
    k = 0
    for item in data[quantity][i]:
        if item == '-':
            data[i][k] = np.nan
        else:
            data[i][k] = float(data[i][k])
        k += 1

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data[['duration','orig_bytes', 'resp_bytes']] = imp.fit_transform(data[['duration','orig_bytes', 'resp_bytes']])

imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
data[data.columns] = imp.fit_transform(data[data.columns])

le = LabelEncoder()

for item in quality:
    data[item]= le.fit_transform(data[item])
    dtype = float

ss = StandardScaler()
data.iloc[:, :-2] = ss.fit_transform(data.iloc[:,:-2])

pca = PCA(svd_solver='full')

datadf = data.copy()

column=[]
for i in range(len(data.columns) -2):
    column.append(f'pc{i+1}')
column.append('label' )
column.append('detailed-label')
datadf.columns = column
datadf.iloc[:,:-2] = pca.fit_transform(datadf.iloc[:, :-2])

print(datadf)

#Рисуем данные в пространстве первых двух главных компонент
plt.figure()
plt.grid()
plt.scatter(datadf['pc1'], datadf['pc2'], c=le.fit_transform(datadf['label']), lw=.6, edgecolors='red')
plt.axis('equal')
plt.title("Данные в пространстве первых двух главных компонент")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()

#Рисуем тепловую карту по координатам первых двух главных компонент
plt.matshow(pca.components_[:2])
plt.colorbar()
plt.gca().xaxis.tick_bottom()
plt.xticks(range(len(datadf.columns) -1), data.iloc[:,:-1], rotation=90)
plt.yticks(range(2), datadf[['pc1','pc2']])
plt.title("Тепловая карта")

plt.show()