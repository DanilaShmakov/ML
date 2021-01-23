import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

no_skin_study =['NoSkin.jpg','NoSkin2.jpg','NoSkin3.jpg']
skin_study =['Skin.png','Skin2.png','Skin3.png','grad.png']
test_photos=['man.jpg', 'Putin3.jpg', 'big.jpg']
gnb = GaussianNB()
ss = StandardScaler()
#RGB

df_skin_rgb = pd.DataFrame(columns=['R','G','B']).drop_duplicates().reset_index(drop=True)
df_no_skin_rgb = pd.DataFrame(columns=['R','G','B']).drop_duplicates().reset_index(drop=True)

for item in skin_study:
    skin_rgb = Image.open(item).convert('RGB')
    skin_colors_rgb = np.array(skin_rgb).reshape(skin_rgb.size[0]*skin_rgb.size[1],3)
    df_skin_rgb_temp = pd.DataFrame(skin_colors_rgb, columns=['R', 'G', 'B']).drop_duplicates().reset_index(drop=True)
    df_skin_rgb = pd.concat([df_skin_rgb, df_skin_rgb_temp]).reset_index(drop=True)

for item in no_skin_study:
    no_skin_rgb = Image.open(item).convert('RGB')
    no_skin_colors_rgb = np.array(no_skin_rgb).reshape(no_skin_rgb.size[0]*no_skin_rgb.size[1],3)
    df_no_skin_rgb_temp = pd.DataFrame(no_skin_colors_rgb, columns=['R', 'G', 'B']).drop_duplicates().reset_index(drop=True)
    df_no_skin_rgb = pd.concat([df_no_skin_rgb, df_no_skin_rgb_temp]).reset_index(drop=True)

ones = pd.DataFrame(1, index=np.arange(len(df_skin_rgb)), columns=['label'])
zeros = pd.DataFrame(0, index=np.arange(len(df_no_skin_rgb)), columns=['label'])

df_skin_rgb = df_skin_rgb.assign(label=ones)
df_no_skin_rgb = df_no_skin_rgb.assign(label=zeros)

df_rgb = pd.concat([df_skin_rgb,
                    df_no_skin_rgb.loc[np.random.choice(df_no_skin_rgb.index, len(df_skin_rgb)
                                                        ,replace=False)]]).reset_index(drop=True)

# нормирование данных
# df_rgb.iloc[:, :-1] = ss.fit_transform(df_rgb.iloc[:, :-1])

print(f'Кожа/не кожа в RGB: {len(df_rgb[df_rgb["label"]==1])} {len(df_rgb[df_rgb["label"]==0])}')


points_train_rgb, points_test_rgb, labels_train_rgb, labels_test_rgb = train_test_split(df_rgb.iloc[:, :-1],
    df_rgb['label'], test_size=0.25, random_state=0)
gnb.fit(points_train_rgb, labels_train_rgb)
prediction_hsv = gnb.predict(points_test_rgb)

#оценка RGB

print("Оценка RGB: ", format(gnb.score(points_test_rgb, labels_test_rgb)))

scores=cross_val_score(gnb, df_rgb[df_rgb.columns[:-1]], df_rgb["label"], cv=10)
print(scores)
#тест классификатора

for item in test_photos:
    me_rgb = Image.open(item).convert('RGB')
    data_rgb = np.array(me_rgb).reshape(me_rgb.size[0]*me_rgb.size[1], 3)
    df_rgb_test = pd.DataFrame(data_rgb, columns=['R','G','B'])
# df_rgb_test.iloc[:, :] = ss.fit_transform(df_rgb_test.iloc[:, :])
    prediction = gnb.predict(df_rgb_test)
    img = np.array(me_rgb)
    for i in range(len(img)):
        n = len(img[i])
        for j in range(n):
            if prediction[i * n + j] == 1:
                img[i][j]=(255,0,0)

    Image.fromarray(img, mode="RGB").show()



#HSV


df_skin = pd.DataFrame(columns=['H','S','V']).drop_duplicates().reset_index(drop=True)
df_no_skin = pd.DataFrame(columns=['H','S','V']).drop_duplicates().reset_index(drop=True)

for item in skin_study:
    skin_hsv = Image.open(item).convert('HSV')
    skin_colors_hsv = np.array(skin_hsv).reshape(skin_hsv.size[0]*skin_hsv.size[1],3)
    df_skin_temp = pd.DataFrame(skin_colors_hsv, columns=['H', 'S', 'V']).drop_duplicates().reset_index(drop=True)
    df_skin = pd.concat([df_skin, df_skin_temp]).reset_index(drop=True)

for item in no_skin_study:
    no_skin_hsv = Image.open(item).convert('HSV')
    no_skin_colors_hsv = np.array(no_skin_hsv).reshape(no_skin_hsv.size[0]*no_skin_hsv.size[1],3)
    df_no_skin_temp = pd.DataFrame(no_skin_colors_hsv, columns=['H', 'S', 'V']).drop_duplicates().reset_index(drop=True)
    df_no_skin = pd.concat([df_no_skin, df_no_skin_temp]).reset_index(drop=True)


ones = pd.DataFrame(1, index=np.arange(len(df_skin)), columns=['label'])
zeros = pd.DataFrame(0, index=np.arange(len(df_no_skin)), columns=['label'])

df_skin = df_skin.assign(label=ones)
df_no_skin = df_no_skin.assign(label=zeros)

df_hsv = pd.concat([df_skin,
                    df_no_skin.loc[np.random.choice(df_no_skin.index, len(df_skin), replace=False)]]).reset_index(drop=True)

# нормирование данных
# df_hsv.iloc[:, :-1] = ss.fit_transform(df_hsv.iloc[:, :-1])


print(f'Кожа/не кожа в HSV: {len(df_hsv[df_hsv["label"]==1])} {len(df_hsv[df_hsv["label"]==0])}')

points_train_hsv, points_test_hsv, labels_train_hsv, labels_test_hsv = train_test_split(df_hsv.iloc[:, :-1],
    df_hsv['label'], test_size=0.25, random_state=0)
gnb.fit(points_train_hsv, labels_train_hsv)
prediction_hsv = gnb.predict(points_test_hsv)

#оценка HSV

print("Оценка HSV: ", format(gnb.score(points_test_hsv, labels_test_hsv)))
scores=cross_val_score(gnb, df_hsv[df_hsv.columns[:-1]], df_hsv["label"], cv=10)
print(scores)

#тест классификатора
for item in test_photos:
    me_hsv = Image.open(item).convert('HSV')
    data_hsv = np.array(me_hsv).reshape(me_hsv.size[0]*me_hsv.size[1], 3)
    df_hsv_test = pd.DataFrame(data_hsv, columns=['H','S','V'])
    # df_hsv_test.iloc[:, :] = ss.fit_transform(df_hsv_test.iloc[:, :])
    # print(df_hsv_test.loc[np.random.choice(df_hsv_test.index, 5, replace=False)])
    prediction = gnb.predict(df_hsv_test)
    img = np.array(me_hsv)

    for i in range(len(img)):
        n = len(img[i])
        for j in range(n):
            if prediction[i * n + j] == 1:
                img[i][j]=(0,255,255)

    Image.fromarray(img, mode="HSV").show()
