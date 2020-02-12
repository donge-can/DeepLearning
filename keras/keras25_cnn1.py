from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(7, (2,2) , padding = 'valid', strides = 2,
                 input_shape = (5, 5, 1)))

# padding = 'same' : input 차원과 동일한 의미
# padding = 'valid' : 패딩 적용 안함
# padding : 과적합 방지, 데이터의 손실 최소화하기 위해서

model.add(Conv2D(8, (2,2), padding = 'same'))

model.add(MaxPooling2D(pool_size=(2,2), strides= (1,1)))
# stride : 해당 간격 / 만약 none 값이면, pool_size 를 그대로 받아서 입력함
# pool_size : 최대로 자를 개수
model.add(Flatten())
model.add(Dense(1))
model.summary()