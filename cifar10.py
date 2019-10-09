import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets.cifar10 import load_data
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 10

(X_train, y_train), (X_test, y_test) = load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

num_layers = [1, 2, 3]
num_neurons = [128, 256, 512]
dropouts = [0.0, 0.1, 0.2]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

best_acc = 0.000
best_layer_amt = 0
best_neuron_amt = 0
best_dropout = 0

# for i in num_layers:
#     for j in num_neurons:
#         for k in dropouts:
#             model = Sequential()
#
#             # model.add(Flatten(input_shape=(32, 32)))
#
#             model.add(Conv2D(j, kernel_size=(5, 5), input_shape=(X_train.shape[1:])))
#             model.add(Activation('relu'))
#             model.add(GlobalAveragePooling2D())
#             model.add(Dropout(k))
#
#             if i == 2:
#                 model.add(Conv2D(j, kernel_size=(3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(GlobalAveragePooling2D())
#                 model.add(Dropout(k))
#             elif i == 3:
#                 model.add(Conv2D(j, kernel_size=(3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(GlobalAveragePooling2D())
#                 model.add(Dropout(k))
#
#                 model.add(Conv2D(j, kernel_size=(3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(GlobalAveragePooling2D())
#                 model.add(Dropout(k))
#
#             model.add(Dense(10))
#             model.add(Activation('softmax'))
#
#             model.compile(loss='sparse_categorical_crossentropy',
#                           optimizer='Adam',
#                           metrics=['accuracy'])
#
#             model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
#
#             (val_loss, val_acc) = model.evaluate(X_test, y_test)
#
#             if val_acc > best_acc:
#                 print('New best accuracy found')
#                 best_acc = val_acc
#                 best_layer_amt = i
#                 best_neuron_amt = j
#                 best_dropout = k
#
# print(f'Best amount of layers: {best_layer_amt}')
# print(f'Best neuron amount: {best_neuron_amt}')
# print(f'Best dropout rate: {best_dropout}')
# print(f'Best accuracy: {best_acc}')

model = Sequential()

# First convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Second convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)
