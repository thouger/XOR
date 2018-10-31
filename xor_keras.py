import numpy as np
from keras import Sequential
from keras.layers import Dense

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0, 1, 1, 0]])
model = Sequential()
model.add(Dense(32,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
model.fit(X,Y.T, epochs=2000, verbose=2)

print(model.predict(X))