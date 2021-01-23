import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

skin = Image.open('Skin.png').convert('RGB')
dataskin = np.array(skin)
dataskin = dataskin.reshape(skin.size[0] * skin.size[1], 3)

noskin = Image.open('NoSkin3.jpg').convert('RGB')
datanoskin = np.array(noskin)
datanoskin = datanoskin.reshape(noskin.size[0] * noskin.size[1], 3)

listBack = [255,255,255]
listR=[255]
listG=[255]
listB=[255]

listLabel=[0]

for item in dataskin:
    if (item != listBack).any():
        listR.append(item.tolist()[0])
        listG.append(item.tolist()[1])
        listB.append(item.tolist()[2])
        listLabel.append(1)

for item in datanoskin:
    listR.append(item.tolist()[0])
    listG.append(item.tolist()[1])
    listB.append(item.tolist()[2])
    listLabel.append(0)

df = pd.DataFrame({'R':listR,'G':listG,'B':listB, 'label':listLabel})
# print(df)

df = df.drop_duplicates(subset=['R', 'G', 'B'], keep=False)
# print(df)

print(f'Кожа/не кожа в RGB: {len(df[df["label"]==1])} {len(df[df["label"]==0])}')

points_train, points_test, labels_train, labels_test = train_test_split(df.iloc[:, :-1], df['label'],
                                                                       test_size=0.25,random_state=0)
gnb = GaussianNB()
gnb.fit(points_train, labels_train)
prediction = gnb.predict(points_test)

print(f'Оценка RGB: {format(gnb.score(points_test, labels_test))}')
scores=cross_val_score(gnb, df[df.columns[:-1]], df["label"], cv=10)
print(scores)

replacement_color = (255,0,0)

test = Image.open('man.jpg').convert('RGB')
datatest = np.array(test)
datatest = datatest.reshape(test.size[0]*test.size[1],3)

listpredict = gnb.predict(datatest.reshape(test.size[0]*test.size[1],3))

img = np.array(test)
row = 0
column = 0
for item in img:
    for i in item:
        if listpredict[row* len(item) + column] == 1:
            img[row][column] = replacement_color
        column += 1
    column=0
    row+=1

img2 = Image.fromarray(img, mode='RGB')
img2.show()

#HSV



skin_hsv = Image.open('Skin.png').convert('HSV')
no_skin_hsv = Image.open('NoSkin3.jpg').convert('HSV')

#перевод данных в удобный формат
skin_colors_hsv = np.array(skin_hsv).reshape(skin_hsv.size[0]*skin_hsv.size[1],3)
no_skin_colors_hsv = np.array(no_skin_hsv).reshape(no_skin_hsv.size[0]*no_skin_hsv.size[1],3)

#создание датафреймов
df_skin = pd.DataFrame(skin_colors_hsv, columns=['H','S','V']).drop_duplicates().reset_index(drop=True)
df_no_skin = pd.DataFrame(no_skin_colors_hsv, columns=['H','S','V']).drop_duplicates().reset_index(drop=True)

ones = pd.DataFrame(1, index=np.arange(len(df_skin)), columns=['label'])
zeros = pd.DataFrame(0, index=np.arange(len(df_no_skin)), columns=['label'])

df_skin = df_skin.assign(label=ones)
df_no_skin = df_no_skin.assign(label=zeros)


df_hsv = pd.concat([df_skin, df_no_skin]).reset_index(drop=True)

# ss = StandardScaler()
# df_hsv.iloc[:, :-1] = ss.fit_transform(df_hsv.iloc[:, :-1])

# print(df_skin)
print(f'Кожа/не кожа в HSV: {len(df_hsv[df_hsv["label"]==1])} {len(df_hsv[df_hsv["label"]==0])}')

points_train_hsv, points_test_hsv, labels_train_hsv, labels_test_hsv = train_test_split(df_hsv.iloc[:, :-1],
    df_hsv['label'], test_size=0.25, random_state=0)
gnb.fit(points_train_hsv, labels_train_hsv)
prediction_hsv = gnb.predict(points_test_hsv)

#оценка HSV
print("Оценка HSV: ", format(gnb.score(points_test_hsv, labels_test_hsv)))
# scores=cross_val_score(gnb, df_hsv[df_hsv.columns[:-1]], df_hsv["label"], cv=10)
# print(scores)
#тест классификатора
me_hsv = Image.open('man.jpg').convert('HSV')
data_hsv = np.array(me_hsv).reshape(me_hsv.size[0]*me_hsv.size[1], 3)

prediction = gnb.predict(data_hsv)

img = np.array(me_hsv)

for i in range(len(img)):
    n = len(img[i])
    for j in range(n):
        if prediction[i * n + j] == 1:
            img[i][j]=(0,255,255)

Image.fromarray(img,mode="HSV").show()



#
# from skimage.color import rgb2hsv
# import colorsys
#
# listH=[0]
# listS=[0]
# listV=[255]
# listLabelHSV=[0]
#
# skinHSV = Image.open('Skin.png').convert('RGB')
# dataskinHSV = np.array(skinHSV)
# dataskinHSV = dataskinHSV.reshape(skinHSV.size[0] * skinHSV.size[1], 3)
#
# for item in dataskinHSV:
#     if (item != [255,255,255]).any():
#         c2HSV = colorsys.rgb_to_hsv(item[0], item[1], item[2])
#         listH.append(c2HSV[0])
#         listS.append(c2HSV[1])
#         listV.append(c2HSV[2])
#         listLabelHSV.append(1)
#
# noskinHSV = Image.open('NoSkin.jpg').convert('RGB')
# datanoskinHSV = np.array(skinHSV)
# datanoskinHSV = datanoskinHSV.reshape(skinHSV.size[0] * skinHSV.size[1], 3)
#
# for item in datanoskinHSV:
#     c2HSV = colorsys.rgb_to_hsv(item[0], item[1], item[2])
#     listH.append(c2HSV[0])
#     listS.append(c2HSV[1])
#     listV.append(c2HSV[2])
#     listLabelHSV.append(0)
#
#
# dfHSV = pd.DataFrame({'H':listH, 'S':listS, 'V':listV, 'label':listLabelHSV})
#
# # dfHSV = pd.DataFrame(columns=['H','S','V'], data = dataskinHSV)
# print(dfHSV)
# # dfHSV = dfHSV.drop(dfHSV[(dfHSV['H']==0) & (dfHSV['S']==0) & (dfHSV['V']==255)], axis=1)
# # print(dfHSV)
# dfHSV = dfHSV.drop_duplicates()
# print(dfHSV)
#
# points_train, points_test, labels_train, labels_test = train_test_split(dfHSV.iloc[:, :-1], dfHSV['label'],
#                                                                        test_size=0.25,random_state=0)
# gnbHSV = GaussianNB()
# gnbHSV.fit(points_train, labels_train)
# prediction = gnbHSV.predict(points_test)
#
# print(format(gnbHSV.score(points_test, labels_test)))
#
# test = Image.open('Putin3.jpg').convert('RGB')
# datatestHSV = np.array(test)
# datatestHSV = datatestHSV.reshape(test.size[0]*test.size[1],3)
#
# listdtHSV = []
# for item in datatestHSV:
#     listdtHSV.append(colorsys.rgb_to_hsv(item[0], item[1],item[2]))
#
# print(listdtHSV)
# listpredictHSV = gnbHSV.predict(listdtHSV)
#
#
# img = np.array(test)
# row = 0
# column = 0
# for item in img:
#     for i in item:
#         if listpredictHSV[row* len(item) + column] == 1:
#             img[row][column] = replacement_color
#         column += 1
#     column=0
#     row+=1
#
# img2 = Image.fromarray(img, mode='RGB')
# img2.show()