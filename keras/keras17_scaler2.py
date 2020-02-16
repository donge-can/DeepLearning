from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


# 1. Data
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
           [6,7,8],[7,8,9],[8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000],[30000,40000,50000],[40000,50000,60000], [100, 200, 300] ])  
y = array([4,5,6,7,8, 9, 10, 11, 12, 13, 50000, 60000, 70000, 400])





### 실습
# train은 10개, 나머지는 test
# LSTM + LSTM + Dense모델로 구현
# return_sequences 인수를 True로 하면 출력 순서열 중 마지막 값만 출력하는 것이 아니라 전체 순서열을 3차원 텐서 형태로 출력

# 1-1. 데이터 분리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1, shuffle=False)



# 1-2. 데이터 Standardization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.mean(x_train), np.std(x_train))


# LSTM 을 위한 3차원 변경
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)



# 2. model

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1), return_sequences = True))
model.add(LSTM(10, activation = 'relu', input_shape = (3,1), return_sequences = False))   
model.add(Dense(5))
model.add(Dense(1))

model.summary()


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])  


from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience = 20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose =1, callbacks = [early_stopping]) 


# 4. 평가 예측
loss, mae =model.evaluate(x_test, y_test, batch_size=1)
print('loss:', loss)

y_predict = model.predict(x_test, batch_size=1)
print(y_predict)


# R2 : 결정계수
x_prd = array([[250, 260, 270]])
x_prd = scaler.transform(x_prd)

# LSTM 을 위해 차원 변경
x_prd = x_prd.reshape(1, 3, 1)

y_prd = model.predict(x_prd, batch_size=1)
print(y_prd)
