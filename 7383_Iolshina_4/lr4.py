import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from numpy import asarray

EPOCHS = 5
result = dict()

def get_img(filename):
    img = Image.open(filename).convert('RGB')
    img = img.resize((28,28))
    return np.expand_dims((1.0 - (asarray(img) / 255)), axis=0)


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images / 255.0

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))


def train_model(optimizer, epochs):
    optimizerConf = optimizer.get_config()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=128, validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)

    plt.title('Training and test accuracy')
    plt.plot(history.history['accuracy'], 'm', label='train')
    plt.plot(history.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.savefig("./%s%s_acc.png" % (optimizerConf["name"], optimizerConf["learning_rate"]), format='png')
    plt.show()
    plt.clf()

    plt.title('Training and test loss')
    plt.plot(history.history['loss'], 'm', label='train')
    plt.plot(history.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.savefig("./%s%s_loss.png" % (optimizerConf["name"], optimizerConf["learning_rate"]), format='png')
    plt.show()
    plt.clf()

    result["%s%s" % (optimizerConf["name"], optimizerConf["learning_rate"])] = test_acc


for learning_rate in [0.001, 0.01]:
    train_model(optimizers.Adagrad(learning_rate=learning_rate), EPOCHS)
    train_model(optimizers.Adam(learning_rate=learning_rate), EPOCHS)
    train_model(optimizers.RMSprop(learning_rate=learning_rate), EPOCHS)
    train_model(optimizers.SGD(learning_rate=learning_rate), EPOCHS)

for res in result:
    print("%s: %s" % (res, result[res]))