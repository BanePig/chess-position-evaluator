import tensorflow as tf


def create_model():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    net.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics='mae')
    return net
