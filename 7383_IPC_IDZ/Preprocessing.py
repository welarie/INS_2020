import pandas as pd
from nltk.tokenize import TweetTokenizer

fname = f"training.1600000.processed.noemoticon.csv"

cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
df = pd.read_csv(fname, header=None, names=cols, encoding="latin-1")
df.head(5)

# Нас интересует колонка sentiment -- то, что мы предсказываем (target)
# Проверим баланс классов
print(df['sentiment'].value_counts())

# Датасет сбалансирован --> значит хорошей метрикой качества классификации будет accuracy

# Удалим колонки id, date, quety_string и user, т.к. они не предоставляют релевантной информации
df.drop(['id','date','query_string','user'], axis='columns', inplace=True)

# Токенезируем текст
tokenizer = TweetTokenizer()
df['text'] = [tokenizer.tokenize(t.lower()) for t in df['text']]
df.head(5)

# Запишем данные для дальнейшей работы
df.to_csv('train_data.csv', index=False)
