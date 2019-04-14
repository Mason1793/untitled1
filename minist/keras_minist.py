import keras;
import matplotlib.pyplot as plt # 导入可视化的包
import numpy as np
from keras.datasets import mnist # 从keras中导入mnist数据集
from keras.models import Sequential # 导入序贯模型
from keras.layers import Dense # 导入全连接层
from keras.optimizers import SGD # 导入优化函数
import os;
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_data():
    path='mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data() # 下载mnist数据集
# print(x_train.shape, y_train.shape) # 60000张28*28的单通道灰度图
# print(x_test.shape, y_test.shape)

im = plt.imshow(x_train[0], cmap='gray')
# plt.show()
# y_train[0]


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# print(x_train.shape)
# print(x_test.shape)
# print(x_train[0])

x_train = x_train / 255
x_test = x_test / 255
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

model = Sequential() # 构建一个空的序贯模型
# 添加神经网络层
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test)
print("loss:",score[0])
print("accu:",score[1])