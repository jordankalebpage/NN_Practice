import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.datasets.boston_housing import load_data
from tensorflow.keras.models import Sequential

EPOCHS = 100
BATCH_SIZE = 32

(X_train, y_train), (X_test, y_test) = load_data(test_split=0.2)

mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std

X_test -= mean
X_test /= std

# X_train = tf.keras.utils.normalize(X_train)
# X_test = tf.keras.utils.normalize(X_test)

num_layers = [1, 2, 3]
num_neurons = [16, 32, 64]
dropouts = [0.0, 0.2, 0.5]
best_layer_amt = 0
best_neuron_amt = 0
best_dropout = 0
best_mae = 10000

for i in num_layers:
    for k in num_neurons:
        for j in dropouts:
            model = Sequential()

            model.add(Dense(k, input_shape=(X_train.shape[1],)))
            model.add(Activation('relu'))
            model.add(Dropout(j))

            if i == 2:
                model.add(Dense(k))
                model.add(Activation('relu'))
                model.add(Dropout(j))
            elif i == 3:
                model.add(Dense(k))
                model.add(Activation('relu'))
                model.add(Dropout(j))

                model.add(Dense(k))
                model.add(Activation('relu'))
                model.add(Dropout(j))

            model.add(Dense(1))
            model.add(Activation('linear'))

            model.compile(optimizer='Adam',
                          loss='mse',
                          metrics=['mae'])

            model.fit(X_train, y_train,
                      validation_data=(X_test, y_test),
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      verbose=0)

            (val_loss, val_mae) = model.evaluate(X_test, y_test)

            if val_mae < best_mae:
                best_mae = val_mae
                best_layer_amt = i
                best_neuron_amt = k
                best_dropout = j
                print('New best MAE found')

print(f'Best amount of layers: {best_layer_amt}')
print(f'Best neuron amount: {best_neuron_amt}')
print(f'Best dropout rate: {best_dropout}')
print(f'Best MAE: {best_mae}')
