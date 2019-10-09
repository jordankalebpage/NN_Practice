import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

EPOCHS = 10
BATCH_SIZE = 32
MAX_WORDS = 10000


def decode_words(dict_of_elements, value_to_find):
    word = ''
    list_of_items = dict_of_elements.items()
    for item in list_of_items:
        if item[1] == value_to_find:
            word = item[0]
    return word


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.reuters.load_data(test_split=0.2)
word_index = tf.keras.datasets.reuters.get_word_index()

num_classes = max(y_train) + 1

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')

# X_train = X_train.reshape(-1, 1, len(X_train))
# X_test = X_test.reshape(-1, 1, len(X_test))
# y_train = y_train.reshape(len(y_train), 1)
# y_test = y_test.reshape(len(y_test), 1)

model = Sequential()

model.add(Dense(128, input_shape=(MAX_WORDS,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy']
              )

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
