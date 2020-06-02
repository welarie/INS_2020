import numpy as np
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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


# Векторизируем слова (мы не можем работать со строками, только с числами)
# Будем использовать tf-idf
vectorizer = TfidfVectorizer(max_features=5000)

vectorizer.fit(df['text'])
train_x = vectorizer.transform(df_train['text'])
test_x = vectorizer.transform(df_test['text'])


# Создаём классификатор
bayes_classifier = naive_bayes.MultinomialNB()
# Обучим его
bayes_classifier.fit(train_x, train_y)

# Проверим качество
accuracy = accuracy_score(bayes_classifier.predict(test_x), test_y)
print("Качество работы наивного классификатора: {}%".format(round(accuracy * 100)))
