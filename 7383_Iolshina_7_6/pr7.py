from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from var6 import gen_sequence

def gen_data_from_sequence(seq_len = 1000, lookback = 10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i,i+lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback,len(seq))])
    return (past, future)

def gen_data(data, res):
    dataset_size = len(data)
    train_size = (dataset_size // 10) * 7
    val_size = (dataset_size - train_size) // 2

    train_data, train_res = data[:train_size], res[:train_size]
    val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
    test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]
    return train_data, train_res, val_data, val_res, test_data, test_res

def build_model():
    model = Sequential()
    model.add(layers.GRU(32,recurrent_activation='sigmoid',input_shape=(None,1),return_sequences=True))
    model.add(layers.LSTM(32,activation='relu',input_shape=(None,1),return_sequences=True,dropout=0.2))
    model.add(layers.GRU(16,input_shape=(None,1),recurrent_dropout=0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer='nadam', loss='mse')
    return model

def draw_plot():
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(len(loss)), loss, 'm', label='Train')
    plt.plot(range(len(val_loss)), val_loss, 'b', label='Validation')
    plt.grid()
    plt.title('Model loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss.png', format='png', dpi=240)
    plt.clf()

    predicted_res = model.predict(test_data)
    pred_length = range(len(predicted_res))
    plt.plot(pred_length, predicted_res, 'm', label='Predicted')
    plt.plot(pred_length, test_res, 'b', label='Generated')
    plt.title('Sequence')
    plt.xlabel('x')
    plt.ylabel('Sequence')
    plt.grid()
    plt.legend()
    plt.savefig('Sequence.png', format='png', dpi=240)

if __name__ == '__main__':
    data, res = gen_data_from_sequence()
    train_data, train_res, val_data, val_res, test_data, test_res = gen_data(data, res)
    model = build_model()
    history = model.fit(train_data, train_res, epochs=60, validation_data=(val_data, val_res))
    draw_plot()


