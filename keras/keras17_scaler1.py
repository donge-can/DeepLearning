from numpy import array
import numpy as np

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000], [30000,40000,50000], [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

### 실습
# train은 10개, 나머지는 test
# Dense모델로 구현

# 1-1. 데이터 분리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1, shuffle=False)
print(len(x_train))


# 1-2. 데이터 Standardization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)
# # 2. model

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

# model.add(Dense(5, input_dim=1))
model.add(Dense(5, input_shape=(3,)))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
 

#3. Train
model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])
# metrics= ['mse', 'mae', 'rmse', 'rmae']


model.fit(x_train, y_train, epochs=100, batch_size=1)



#4. Evaluation & Prediction
loss, mse =model.evaluate(x_test, y_test, batch_size=1)
print('mse:', mse)

y_predict = model.predict(x_test, batch_size=1)
print(y_predict)


# R2 : 결정계수
from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test, y_predict)
print('R2:', r2_y_predict)



# 다른 예측
x_prd = np.array([[250, 260, 270]])
x_prd = scaler.transform(x_prd)

y_prd = model.predict(x_prd, batch_size= 1)
print(y_prd)


r2_y_predict = r2_score(y_prd, y_predict)
print('R2:', r2_y_predict)
# 여기선 r2 계산 불가함
# 대응되는 y_prd, y_predict 개수가 다름

