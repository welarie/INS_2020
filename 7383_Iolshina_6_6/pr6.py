import numpy as np
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from var6 import gen_data

def loadData():
    X, Y = gen_data(1000)
    X = np.asarray(X)
    Y = np.asarray(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    encoder = LabelEncoder()
    encoder.fit(Y)
    y_test = np.asarray(encoder.transform(y_test))
    y_train = np.asarray(encoder.transform(y_train))
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    x_train = np.asarray(x_train).reshape(x_train.shape[0], 50, 50, 1)
    x_test = np.asarray(x_test).reshape(x_test.shape[0], 50, 50, 1)
    return x_train, x_test, y_train, y_test

def buildModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(50, 50, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=5, strides=1, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = loadData()
    model = buildModel()
    history = model.fit(x_train, y_train, epochs=12, batch_size=100, validation_data=(x_test, y_test))
    ev = model.evaluate(x_test, y_test)

    print("Model accuracy: %s" % (ev[1]))