import numpy as np
from sklearn.impute import SimpleImputer

def check_data(data, trust, quantity):
    for item in quantity:
        print(data[item].to_numpy())
        if ((data[item].to_numpy() >= trust[item][0]) and
           (data[item].to_numpy() <= trust[item][1])):
            print(f'{item} в пределах доверительного интервала')
        else:
            print(f'За пределами{item}')
            return False
    return True


def normalize(data, quantity):
    for i in quantity:
        k = 0
        for item in data[i]:
            if item == '-':
                data[i][k] = np.nan
            k += 1

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(data[quantity])
    data[quantity] = imp.transform(data[quantity])
    # print('Количество пропусков в:\n',data.isnull().sum())
    return data

