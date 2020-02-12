from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) #(60000, 28, 28)
print(y_train.shape) #(60000,)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

# print(x_train)


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) # (60000, 10)


model = Sequential()
model.add(Conv2D(7, (2,2) , padding = 'valid', strides = 2,
                 input_shape = (28, 28, 1)))

model.add(Conv2D(8, (2,2), padding = 'same'))

model.add(MaxPooling2D(pool_size=(2,2), strides= (1,1)))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
# 다중 분류 : softmax

model.summary()


model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

early_stopping = EarlyStopping(monitor='loss', patience = 20, mode='auto')
model.fit(x_train, y_train, validation_split = 0.2, batch_size=8, epochs=100, callbacks=[early_stopping])

acc = model.evaluate(x_test, y_test)
print(acc)
# [0.21186635724841812, 0.975600004196167]