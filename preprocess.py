import data
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

train_sentences = data.clean_sentences[:7000]
validation_sentences = data.clean_sentences[7000:]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sentences)
tokenizer.fit_on_texts(validation_sentences)

train_sequences = tokenizer.texts_to_sequences(
    train_sentences
)
validation_sequences = tokenizer.texts_to_sequences(
    validation_sentences
)

train_padded = pad_sequences(train_sequences, maxlen=120, padding='pre', truncating='pre')
validation_padded = pad_sequences(validation_sequences, maxlen=120, padding='pre', truncating='pre')

vocab_size = len(tokenizer.word_index)
print(vocab_size)
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

y_train = []
y_validate = []
for item in range(3724):
  y_train.append(item)
for item in range(199):
  y_validate.append(item)

