from numpy import array
import numpy as np


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        print(end_ix)
        if end_ix > len(sequence)-1 :
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        
    return array(X), array(y)



dataset = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_steps = 3



x, y = split_sequence(dataset, n_steps)

# for i in range(len(x)):
#     print(x[i], y[i])
    
    
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

# 2. model
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (3,1)))   # (열, 몇 개씩 자를지)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])  
model.fit(x, y, epochs=100, batch_size=1)  


# 4. 평가예측
loss, mae = model.evaluate(x, y, batch_size = 1)
print('loss', loss)


x_input = array([[90,100,110]])
print(x_input.shape)
x_input = x_input.reshape(1,3, 1)

y_predict = model.predict(x_input)
print(y_predict)