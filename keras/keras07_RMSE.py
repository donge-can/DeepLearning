import numpy as np
from sklearn.model_selection import train_test_split

#1. data
x = np.arange(1,101,1)
y = np.arange(1,101,1)

#1.1 dataset 분리 train_test_split - train / test (6:4)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=1, shuffle=False)
# shuffle = False : 순서에 맞춰서
# stratify는 훈련/테스트 데이터들이 원래의 input dataset의 클래스의 비율과 같은 비율을 가지도록 할 것인지 지정 
# 예를 들어 0,1의 클래스가 input dataset에 20:80 비율로 있었다면 훈련 데이터와 테스트 데이터 역시 각각의 클래스가 같은 비율로 있도록 지정 

#1.2 dataset 분리 train_test_split - test / validation  (5:5)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1, shuffle=False)

print('train:', len(x_train))
print('test: ' , len(x_test))
print('val:', len(x_val))

# print(x.shape)
# print(y.shape)



2. model

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
y_predict = model.predict(x_prd, batch_size=1)
print(a)

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test ,y_predict))

print('RMSE:', RMSE(y_test, y_predict))