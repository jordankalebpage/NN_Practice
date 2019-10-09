import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Flatten
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

EPOCHS = 100
BATCH_SIZE = 32

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, shuffle=True)

# class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Normalize the data
X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)

# Tuning hyperparameters
# curr_acc = 0.000
# best_dropout = 0.0
# best_layer_size = 0
# best_layer_amt = 0
#
# layer_amt = [1, 2, 3]
# layer_size = [16, 32, 64, 128]
# dropouts = [0.1, 0.2, 0.33, 0.5]
#
# for i in layer_amt:
#     for j in layer_size:
#         for k in dropouts:
#             model = Sequential()
#
#             model.add(Dense(j, activation='relu', input_shape=(4,)))
#             model.add(Dropout(k))
#             if i >= 2:
#                 model.add(Dense(j, activation='relu'))
#                 model.add(Dropout(k))
#             if i == 3:
#                 model.add(Dense(j, activation='relu'))
#                 model.add(Dropout(k))
#
#             model.add(Dense(3, activation='softmax'))
#
#             model.compile(loss='sparse_categorical_crossentropy',
#                           optimizer='Adam',
#                           metrics=['accuracy'])
#
#             model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
#
#             val_loss, val_acc = model.evaluate(X_test, y_test)
#             if val_acc > curr_acc:
#                 curr_acc = val_acc
#                 best_layer_amt = i
#                 best_layer_size = j
#                 best_dropout = k
#                 print(f'New best Accuracy found.')
#
# print(f'Best accuracy: {curr_acc}')
# print(f'Best layer amount: {best_layer_amt}')
# print(f'Best layer size: {best_layer_size}')
# print(f'Best dropout rate: {best_dropout}')

try:
    model = tf.keras.models.load_model('2-layer-128-neuron-iris.model')
    print('Model loaded')
except OSError:
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(4,)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))

    model.save('2-layer-128-neuron-iris.model')
    print('Model saved')
