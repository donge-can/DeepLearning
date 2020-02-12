import numpy as np
from sklearn.model_selection import train_test_split

#1. data
x = np.array([range(1,101), range(101,201), range(301,401)])
y = np.array([range(101,201)])
y2 = np.array(range(101,201))
x = np.transpose(x)
y = np.transpose(y)

print(x.shape) # (3,100)
print(y.shape) #(1,100)
print(y2.shape) # (100,)


#1.1 dataset 분리 train_test_split - train / test (6:4)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=1, shuffle=False)
print(x_train)
#1.2 dataset 분리 train_test_split - test / validation  (5:5)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1, shuffle=False)



# 2. model

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=3))
model.add(Dense(32, input_shape=(3,)))
model.add(Dense(18))
model.add(Dense(1))

model.summary()
 

#3. Train
model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))


#4. Evaluation & Prediction
loss, mse =model.evaluate(x_test, y_test, batch_size=1)
print('mse:', mse)

y_predict = model.predict(x_test, batch_size=1)


# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test ,y_predict))

print('RMSE:', RMSE(y_test, y_predict))


# R2 : 결정계수
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print('R2:', r2_y_predict)

print('예측결과:', y_predict)