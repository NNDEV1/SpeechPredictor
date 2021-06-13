import preprocess
from keras.models import Sequential, Model
from keras.layers import Embedding
from keras.layers import LSTM, Activation, Dense, Permute, Dropout, add, dot, concatenate, Bidirectional, GRU

model = Sequential([
    Embedding(preprocess.vocab_size+1, 50),
    GRU(256, return_sequences=True),
    GRU(512, return_sequences=False),
    Dense(100, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(preprocess.train_padded, np.array(preprocess.y_train), batch_size=32, epochs=100)
