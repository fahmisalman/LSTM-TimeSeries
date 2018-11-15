import numpy as np

from tensorflow import keras


def train(x, y, input_dim, epoch):
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_dim=input_dim, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(np.array(x), np.array(y), epochs=epoch, batch_size=2, verbose=2)
    return model


def evaluate(x, y, model):
    mse = model.evaluate(np.array(x), np.array(y), verbose=0)
    return mse


def save_model(model, s):
    model_json = model.to_json()
    with open("model/%s.json" % s, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model/%s.h5" % s)


def load_model(s):
    model_json = open('model/%s.json' % s, 'r').read()
    model = keras.models.model_from_json(model_json)
    model.load_weights("model/%s.h5" % s)
    return model


def predict(x, model):
    return model.predict(np.array(x))
