import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from sklearn.impute import SimpleImputer
import funcs

data = pd.read_csv('IoT.csv', delimiter=',')
quantity = ['ts', 'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts']
data = pd.DataFrame(data, columns=['ts', 'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts',
                                    'resp_pkts', 'proto', 'conn_state', 'label'])
#print(data[k])

funcs.normalize(data, quantity)
# print(data)

plt.figure()
for item in quantity:
    sns.distplot(data[item])
    plt.savefig(f'{item}.pdf')
    # plt.show()

    x = data[item]

    mindata = sns.distplot(x).get_lines()[0].get_data()
    plt.clf()
    minIndex = argrelextrema(mindata[1], np.less)
    minimums = mindata[0][minIndex]
    print(f'{item}')
    print(f'Минимумы = {minimums}')
    print(f'Матожидание: {np.mean(x)}')
    intervals = [()]
    intervals[0] = (0, round(minimums[0], 2))
    for i in range(len(minimums)-1):
        intervals.append((round(minimums[i], 2), round(minimums[i+1], 2)))
    intervals.append((round(minimums[len(minimums)-1], 2), max(x)))
    print(f'Интервалы: {intervals}')
    print(f'Кол-во интервалов: {len(intervals)}')

    trust = []
    for i in intervals:
        mean = np.mean(i)
        std = np.std(i)
        trust.append((round(mean, 2), (round(mean-3*std, 2), round((mean+3*std), 2))))
    print("Области однородности:")
    for ml in trust:
        print(f"μ={ml[0]}; доверительный интервал={ml[1]}")


quality = ['proto', 'conn_state', 'label']

for item in quality:
    x = data[item]
    x.value_counts().plot.bar()
    plt.show()


