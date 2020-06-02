import pandas as pd
import numpy as np
import keras
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras.optimizers import Adam
from datetime import datetime
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

fname = f"train_data.csv"

cols = ['sentiment', 'text']
df = pd.read_csv(fname, header=None, names=cols, encoding="latin-1")
df.head(5)

# Установим random_seed для воспроизводимости экспериментов
SEED = 666
np.random.seed(SEED)

# Разобьём данные на train и test
df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

# Перепишем target в формат 0, 1 <--> негативный, позитивный
train_y = np.array(list(map(lambda x: 1 if x == 4 else 0, df_train['sentiment'])))
test_y = np.array(list(map(lambda x: 1 if x == 4 else 0, df_test['sentiment'])))

# Токенезируем текст
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(df_train['text'])

vocab_size = len(tokenizer.word_index) + 1

# Выровняем все тексты до одного размера
maxlen = 50
train_x = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=maxlen, padding="post")
test_x = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=maxlen, padding="post")


# Строим сеть
def build_model(max_len=maxlen, emb_dim=128, dropout_proba=0.1, gru_state_size=128, bidirectional=False):
    
    model = Sequential()
    model.add(Embedding(len(tokenizer.index_word) + 1, emb_dim, trainable=True, input_length=max_len))

    if bidirectional:
        model.add(Bidirectional(LSTM(gru_state_size)))
    else:
        model.add(LSTM(gru_state_size))

    model.add(Dropout(dropout_proba))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam
    opt = optimizer(lr=1e-3)
    loss = "binary_crossentropy"

    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
    return model

# Обучим bidirectional модель
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [ReduceLROnPlateau(monitor="val_acc"),  # Уменьшаем lr, если выходим на плато
             EarlyStopping(monitor="val_loss", restore_best_weights=True),  # Регуляризация -- метод ранней остановки
             keras.callbacks.TensorBoard(log_dir=logdir),  # Логгирование
             ]

model = build_model(bidirectional=True)
history = model.fit(train_x, train_y, epochs=2, batch_size=128, callbacks=callbacks)

# Сохраним веса используя формат `checkpoint_path` format
checkpoint_path = "bidir.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.save_weights(checkpoint_path)

# Проверяем качество
_, acc = model.evaluate(test_x, test_y, verbose=2)
print("Точность на тестовом множестве: {:5.2f}%".format(round(100 * acc)))
