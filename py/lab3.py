import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import funcs



data = pd.read_csv('IoT.csv', delimiter=',')
labels = pd.DataFrame(data, columns=['label'])
data = pd.DataFrame(data, columns=['duration', 'orig_bytes', 'orig_pkts', 'proto', 'resp_bytes', 'conn_state',
                               'resp_pkts'])

quantity = ['duration', 'orig_bytes', 'orig_pkts', 'resp_bytes', 'resp_pkts']

data = funcs.normalize(data, quantity)

data = pd.get_dummies(data, columns=['proto', 'conn_state'])
dtype = float;
# print(data)

trust_old = dict()

for index in quantity:
     trust_old[index] = [np.mean(data[index]) - (3 * np.std(data[index])),
                 np.mean(data[index]) + (3 * np.std(data[index]))]
     data.loc[(data[index] > trust_old[index][1]) | (data[index] < trust_old[index][0]), index] = np.nan

data = data.dropna()

le = LabelEncoder()
le.fit(labels.label)
labels['label'] = le.transform(labels.label)
data = data.assign(label=labels['label'])

#нормализация данных

ss = StandardScaler()
data.iloc[:, :-1] = ss.fit_transform(data.iloc[:, :-1])
trust = dict()
for index in quantity:
    trust[index] = [np.mean(data[index]) - (3 * np.std(data[index])),
                    np.mean(data[index]) + (3 * np.std(data[index]))]

points_train, points_test, labels_train, labels_test = train_test_split(data.iloc[:, :-1], data['label'],
                                                                        test_size=0.25,random_state=0)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(points_train, labels_train)
prediction = knn.predict(points_test)

test_point = {'duration': [2.71], 'orig_bytes': [3], 'orig_pkts': [3],  'resp_bytes': [2.7], 'resp_pkts': [3],
          'proto_icmp': [0], 'proto_udp': [0], 'proto_tcp': [1], 'conn_state_S0': [0],
          'conn_state_SF': [0], 'conn_state_REJ': [1], 'conn_state_OTH': [0], 'conn_state_RSTR': [0]}

data_new = pd.DataFrame(test_point)

ss.fit(data.iloc[:, :-1])
data_new.iloc[:, :] = ss.transform(data_new.iloc[:, :])

if funcs.check_data(data_new, trust, quantity):
    prediction = knn.predict(data_new)
    data_new = data_new.assign(label=prediction)
    data = data.append(data_new, ignore_index=True)

print('ОЦЕНКА: ', format(knn.score(points_test, labels_test)))
