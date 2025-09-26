import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 1. Load and preprocess dataset
# Ensure you have 'text' and 'label' columns
data = pd.read_csv('path_to_dataset.csv')

texts = data['text'].astype(str).tolist()
labels = data['label'].tolist()

# 2. Tokenize text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
maxlen = 150
X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(labels)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Model definition
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=maxlen),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train
model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=5, batch_size=32)

# 6. Predict sentiment for new text


def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)[0][0]
    return 'Positive' if pred >= 0.5 else 'Negative'


print(predict_sentiment("I love this! An amazing update."))
print(predict_sentiment("This is the worst app ever."))
