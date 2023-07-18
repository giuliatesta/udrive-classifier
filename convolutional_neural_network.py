from keras import Sequential
from keras.src.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow as tf

from preprocessing import WINDOW_SIZE

NUM_CLASSES = 4


# define the CNN model
def define():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=(WINDOW_SIZE, 6)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    adam_optimizer = tf.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=adam_optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# def k_fold_cross_validation():
