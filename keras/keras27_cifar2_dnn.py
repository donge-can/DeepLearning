from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000, 32, 32, 3)
print(y_train.shape) #(50000, 1)

x_train = x_train.reshape(-1, 32*32*3).astype('float32')/255
x_test = x_test.reshape(-1, 32*32*3).astype('float32')/255

print(x_train.shape)


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) # (50000, 10)



from keras.models import Sequential
from keras.layers import Dense



model = Sequential()
model.add(Dense(200, activation = 'relu', input_shape = (32*32*3,)))
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['acc']) 

model.fit(x_train,y_train, 
          epochs=5, batch_size = 1) 


# 평가예측
loss = model.evaluate(x_test, y_test, batch_size = 1)
print('loss', loss)

loss [0.180000002682209, 0.10000000149011612]
