import numpy as np

#1. data
x = np.arange(1,11,1)
y = np.arange(1,11,1)

print(x.shape) #(10,) 10개의 스칼라
print(y.shape) #(10,) 10개의 스칼라


#2. model

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# model.add(Dense(8, input_shape=(1,), activation ='relu'))
model.add(Dense(8, input_dim=1, activation='relu'))
# 첫번째 Dense가 입력층과 은닉층의 역할을 겸함
# 데이터에서 1개의 값을 받아 은닉층의 8개 노드로 보낸다는 뜻

model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# activation : 다음 층으로 어떻게 값을 넘길지 결정 / relu, sigmoid, softmax


#3. Train
model.compile(loss='mse', optimizer = 'adam', metrics = ['mse'])

# 모델 수행 결과 metrics
# loss : 오차값을 추정
# 평균 제곱 계열 - [mse, mae, rmse, rmae] / 교차 엔트로피 계열 - [categorical-crossentropy, binary_crossentropy]
# optimizer : 오차를 어떻게 줄여 나갈지

model.fit(x,y, epochs=100, batch_size=1)
# batch_size default = 32 샘플을 한 번에 몇 개씩 처리할지
# 1000개의 데이터가 있을 때, 1 epoch = 10(batch_size) * 100(step or iteration)


#4. Evaluation & Prediction
loss, mse =model.evaluate(x, y, batch_size=1)
print('mse:', mse)


x_prd = np.arange(11,14,1)
a = model.predict(x_prd, batch_size=1)
print(a)

b = model.predict(x, batch_size=1)
print(b)