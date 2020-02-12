import numpy as np

#1. data
x = np.arange(1,11,1)
y = np.arange(1,11,1)

# print(x.shape)
# print(y.shape)


# 2. 모델 load

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense


model = load_model('./save/savetest01.h5')

# inp = model.input
# out = model.output
# print(out)

# print('끝')

model.add(Dense(5,name='dense_5'))
model.add(Dense(1, name = 'dense_6'))



model.summary()



 
# #3. Train
# model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])
# # metrics= ['mse', 'mae', 'rmse', 'rmae']


# model.fit(x,y, epochs=500, batch_size=1, callbacks = [checkpointer])
# # batch_size default = 32
# # 1000개의 데이터가 있을 때, 1 epoch = 10(batch_size) * 100(step or iteration)


# #4. Evaluation & Prediction
# loss, mse =model.evaluate(x, y, batch_size=1)
# print('mse:', mse)


# x_prd = np.arange(11,14,1)
# a = model.predict(x_prd, batch_size=1)
# print(a)

# b = model.predict(x, batch_size=1)
# print(b) 
