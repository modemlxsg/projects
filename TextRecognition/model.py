import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def VGG(inputs):

    x = layers.Conv2D(64, (3, 3), strides=(
        1, 1), padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)
    x = layers.Conv2D(128, (3, 3), strides=(
        1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)
    x = layers.Conv2D(256, (3, 3), strides=(
        1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(
        1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)
    x = layers.Conv2D(512, kernel_size=2, activation='relu')(x)

    return x


def CRNN(nclass, backbone='vgg', num_rnn=1):
    inputs = layers.Input(shape=(32, None, 1))

    if backbone == 'vgg':
        x = VGG(inputs)
    else:
        raise Exception('backbone error !')

    assert x.shape[1] == 1
    x = tf.squeeze(x, axis=1)

    for _ in range(num_rnn):
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dense(nclass)(x)

    return keras.Model(inputs=inputs, outputs=x, name='CRNN')
