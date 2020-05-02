import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras import layers
from keras.utils import to_categorical
from keras import models
from keras.datasets import imdb

dim = 10000
filename = 'text.txt'

def vectorize(sequences, dimension = dim):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def load_text(filename):
    text = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            text += [s.strip(''.join(['.', ',', ':', ';', '!', '?', '(', ')'])).lower() for s in line.strip().split()]
    print(text)
    index = imdb.get_word_index()
    words = []
    for s in text:
        if s in index and index[s] < 10000:
            words.append(index[s])
    test_text(np.array(words))

def test_text(text):
    text = vectorize([text])
    print(text)
    model = test_model()
    prediction = model.predict(text)
    print(prediction)

def test_model():
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dim)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]

    model = Sequential()
    model.add(layers.Dense(50, activation = "relu", input_shape=(dim, )))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dense(1, activation = "sigmoid"))

    model.compile( optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    history = model.fit(train_x, train_y, epochs= 2, batch_size = 500, validation_data = (test_x, test_y))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'm', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'm', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    results = model.evaluate(test_x, test_y)
    print(results)
    return model

load_text(filename)
#test_model()
