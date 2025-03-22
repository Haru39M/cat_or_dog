# ライブラリのインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
# from google.colab import files
from keras import optimizers
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# exit()

# Googleドライブのファイルパスを指定する。
data_set_dir_cats = "test_set/cats/"
data_set_dir_dog = "test_set/dogs/"

path_cats = os.listdir(data_set_dir_cats)
path_dogs = os.listdir(data_set_dir_dog)

# 画像を格納するリストの作成
img_cats = []
img_dogs = []

# 各カテゴリの画像サイズを変換してリストに保存
img_size = 100  # 画像サイズ：100×100
for i in range(len(path_cats)):
    img = cv2.imread(data_set_dir_cats + path_cats[i])
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = cv2.resize(img, (img_size, img_size))
    img_cats.append(img)

for i in range(len(path_dogs)):
    img = cv2.imread(data_set_dir_dog + path_dogs[i])
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = cv2.resize(img, (img_size, img_size))
    img_dogs.append(img)

# 犬と猫の画像を一つのリストに集約、及び正解ラベルを設定する。
X = np.array(img_cats + img_dogs)
y = np.array([0]*len(img_cats)  # 0:cats
             + [1]*len(img_dogs))  # 1:dogs

# ラベルをランダムに並び変える。
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]

# 学習データを80%、検証データを20％に分割する。
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]

# 正解ラベルをone-hotの形にする。
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_tensor = Input(shape=(img_size, img_size, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# モデルの定義
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(rate=0.5))
top_model.add(Dense(32, activation='relu'))
top_model.add(Dropout(rate=0.5))
top_model.add(Dense(2, activation='softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

for layer in model.layers[:15]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32,
                    epochs=100, validation_data=(X_test, y_test))

# 画像を受け取り、名称を判別する関数


def pred(img):
    img = cv2.resize(img, (img_size, img_size))
    pred = np.argmax(model.predict(np.array([img])))
    if pred == 0:
        return "cats"
    else:
        return "dogs"


# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.summary()

# resultsディレクトリを作成
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# 重みを保存
model.save(os.path.join(result_dir, 'model_100epoch.h5'))

# pred関数に写真を渡して分類を予測
path_pred = os.listdir("single_prediction/")
for i in range(len(path_pred)):
    img = cv2.imread("single_prediction/" + path_pred[i])
    b, g, r = cv2.split(img)
    my_img = cv2.merge([r, g, b])
    plt.imshow(my_img)
    plt.show()
    print(pred(img))

# accuracyとlossのグラフを描画
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="accuracy", ls="-", marker="o")
plt.plot(history.history["val_accuracy"],
         label="val_accuracy", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="loss", ls="-", marker="o")
plt.plot(history.history["val_loss"], label="val_loss", ls="-", marker="x")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="best")

plt.show()
