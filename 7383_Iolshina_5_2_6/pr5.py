import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential

def generate_dataset(size):
    dataset = np.zeros((size, 6))
    dataset_y = np.zeros(size)
    for i in range(size):
        X = np.random.normal(-5, 10)
        e = np.random.normal(0, 0.3)
        dataset[i, :] = (np.round(-np.power(X, 3) + e), np.round(np.log(np.abs(X)) + e), np.round(np.sin(3*X) + e), np.round(np.exp(X) + e), np.round(X + 4 + e), np.round(X + e))
        dataset_y[i] = np.round(-X + np.sqrt(np.abs(X)) + e)
    return np.round(np.array(dataset), decimals=3), np.array(dataset_y)

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model

def create_models():

    main_input = Input(shape=(6,), name='main_input')
    encoded = Dense(64, activation='relu')(main_input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(6, activation='linear')(encoded)

    input_encoded = Input(shape=(6,), name='input_encoded')
    decoded = Dense(32, activation='relu', kernel_initializer='normal')(input_encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(6, name="out_aux")(decoded)

    predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
    predicted = Dense(64, activation='relu')(predicted)
    predicted = Dense(32, activation='relu')(predicted)
    predicted = Dense(1, name="out_main")(predicted)

    encoded = Model(main_input, encoded, name="encoder")
    decoded = Model(input_encoded, decoded, name="decoder")
    predicted = Model(main_input, predicted, name="regr")

    return encoded, decoded, predicted, main_input

def generate_data():
    x_train, y_train = generate_dataset(300)
    x_test, y_test = generate_dataset(60)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train -= y_mean
    y_train /= y_std
    y_test -= y_mean
    y_test /= y_std
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = generate_data()
    encoded, decoded, full_model, main_input = create_models()

    keras_model = build_model()
    keras_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    History = keras_model.fit(x_train, y_train, epochs=40, batch_size=5, verbose=1, validation_data=(x_test, y_test))

    full_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    History = full_model.fit(x_train, y_train, epochs=40, batch_size=5, verbose=1, validation_data=(x_test, y_test))

    encoded_data = encoded.predict(x_test)
    decoded_data = decoded.predict(encoded_data)
    regr = full_model.predict(x_test)

    pd.DataFrame(np.round(regr, 3)).to_csv("result.csv")
    pd.DataFrame(np.round(x_test, 3)).to_csv("x_test.csv")
    pd.DataFrame(np.round(y_test, 3)).to_csv("y_test.csv")
    pd.DataFrame(np.round(x_train, 3)).to_csv("x_train.csv")
    pd.DataFrame(np.round(y_train, 3)).to_csv("y_train.csv")
    pd.DataFrame(np.round(encoded_data, 3)).to_csv("encoded.csv")
    pd.DataFrame(np.round(decoded_data, 3)).to_csv("decoded.csv")

    decoded.save('decoder.h5')
    encoded.save('encoder.h5')
    full_model.save('full.h5')