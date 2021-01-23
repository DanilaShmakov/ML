import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import funcs
from sklearn import tree

data = pd.read_csv('IoT.csv', delimiter=',')
labels = pd.DataFrame(data, columns=['label'])
data = pd.DataFrame(data, columns=['duration', 'orig_bytes', 'orig_pkts', 'proto', 'resp_bytes', 'conn_state',
                               'resp_pkts'])

quantity = ['duration', 'orig_bytes', 'orig_pkts', 'resp_bytes', 'resp_pkts']

data = funcs.normalize(data, quantity)

# переводим качественные признаки в количественные
le = LabelEncoder()
data['proto'] = le.fit_transform(data['proto'])
data['conn_state'] = le.fit_transform(data['conn_state'])
dtype = float
print(data)

#находим доверительные интервалы для каждого количественного признака и удаляем не попавшие в них значения
trust = dict()

for index in quantity:
     trust[index] = [np.mean(data[index]) - (3 * np.std(data[index])),
                 np.mean(data[index]) + (3 * np.std(data[index]))]
     data.loc[(data[index] > trust[index][1]) | (data[index] < trust[index][0]), index] = np.nan

data = data.dropna()

le.fit(labels.label)
labels['label'] = le.transform(labels.label)
data = data.assign(label=labels['label'])

points_train, points_test, labels_train, labels_test = train_test_split(data.iloc[:, :-1], data['label'],
                                                                        test_size=0.25,random_state=0)

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=4)
dtc.fit(points_train, labels_train)
prediction = dtc.predict(points_test)

test_point = {'duration': [2.71], 'orig_bytes': [20], 'orig_pkts': [3],  'resp_bytes': [21], 'resp_pkts': [3],
              'proto': [1], 'conn_state': [0]}

data_new = pd.DataFrame(test_point)

if funcs.check_data(data_new, trust, quantity):
    prediction = dtc.predict(data_new)
    data_new = data_new.assign(label=prediction)
    data = data.append(data_new, ignore_index=True)

    plt.figure()
    tree.plot_tree(dtc)
    plt.show()

print('ОЦЕНКА: ', format(dtc.score(points_test, labels_test)))
