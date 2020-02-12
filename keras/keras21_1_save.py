import numpy as np

#1. data
x = np.arange(1,11,1)
y = np.arange(1,11,1)

# print(x.shape)
# print(y.shape)


#2. model

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=1)) #  node 5개
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()


model.save('./save/savetest01.h5')
print('끝')


 