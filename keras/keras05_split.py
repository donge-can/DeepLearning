import numpy as np

#1. data
x = np.arange(1,101,1)
y = np.arange(1,101,1)

x_train = x[:60]
y_train = y[:60]

x_test = x[60:80]
y_test = y[60:80]
x_val = x[80:]
y_val = y[80:]


# print(x.shape)
# print(y.shape)


#2. model

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=1)) # layer 4개 / node 5개
model.add(Dense(5, input_shape=(1,)))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
 

#3. Train
model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])
# metrics= ['mse', 'mae', 'rmse', 'rmae']


model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))
# batch_size default = 32
# 1000개의 데이터가 있을 때, 1 epoch = 10(batch_size) * 100(step or iteration)


#4. Evaluation & Prediction
loss, mse =model.evaluate(x_test, y_test, batch_size=1)
print('mse:', mse)


x_prd = np.arange(101,104,1)
a = model.predict(x_prd, batch_size=1)
print(a)

b = model.predict(x_prd, batch_size=1)
print(b) 
