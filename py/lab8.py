import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import math

words = ['ЭЦП', 'целостность', 'отправитель', 'методы', 'SSL', 'сообщение', 'код',
         'модель', 'блочные', 'аудит',
         'симметричная', 'шифрование', 'функция', 'владелец', 'атаки', 'утечки', 'ИБ', 'безопасность', 'стандарт',
         'политика', 'ключ', 'уязвимость', 'канал', 'сеть', 'расшифрование', 'вирусы', 'криптографическая',
         'криптосистема', 'DES', 'конфиденциальность', 'md2', 'кузнечик', 'DSS', 'HTTP',
         'standard', 'ЦП', 'логи', 'вредоносное', 'программное',
         'экран', 'межсетевой', 'угрозы', 'моделирование', 'SHA', 'ис', 'документ', 'домен', 'алгоритм', 'protocol',
         'управление', 'доступность', 'сертификат', 'шифртекст',
         'ассиметричная', 'защита', 'криптографическая', 'владелец',
         'разграничение', 'md5', 'кривых', 'технология', 'хэш', 'взлом', 'файлы', 'данные', 'AES', 'поточные', 'магма',
         'ПО', 'HTTPS', 'подпись', 'закрытый', 'персональные', 'открытый', 'сертификата', 'TCP',
         'крипта', 'взлом', 'md4', 'прямые', 'IPO', 'Алиса', 'Боб']

df = pd.DataFrame(columns=words)

# print(len(words))

k = 0
i = 1
arr = []


df = pd.read_csv('lab8.csv')

df2 = df.copy()
pca = PCA(n_components=2)
df2 = pca.fit_transform(df2)

pcdf = pd.DataFrame(data=df2, columns=['pc1', 'pc2'])
print(pca.components_, 'components')  # !

plt.figure()
plt.grid()
plt.scatter(pcdf['pc1'], pcdf['pc2'], edgecolor='black', lw=.4, cmap='jet')
for i in range(len(pcdf)):
    plt.annotate(i, (pcdf['pc1'][i], pcdf['pc2'][i]))
    plt.arrow(0, 0, pcdf['pc1'][i], pcdf['pc2'][i])
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()


def dis(df, num1, num2):
    dis_ = (df['pc1'][num1 - 1] * df['pc1'][num2 - 1] + df['pc2'][num1 - 1] * df['pc2'][num2 - 1]) / \
               math.sqrt((df['pc1'][num1 - 1] ** 2 + df['pc2'][num1 - 1] ** 2) * (
                       df['pc1'][num2 - 1] ** 2 + df['pc2'][num2 - 1] ** 2))
    print("Расстояние между документами", num1, "и", num2, "равно", dis_)

    return dis_


dis(pcdf, 20, 30)
