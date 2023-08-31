from keras import Sequential
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
import tensorflow as tf
from sklearn.model_selection import KFold

from preprocessing import WINDOW_SIZE

NUM_CLASSES = 4   # 1, 2, 3, 4
K = 5

BATCH_SIZE = 256
EPOCHS = 40


# define the CNN model
def define_cnn():
    print("Using CONVOLUTIONAL NEURAL NETWORK")
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=(WINDOW_SIZE, 6)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation="relu"))  # must be double of the first Conv1D filter
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    #adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)  # performance issues for M1/M2 Macs
    adam_optimizer = tf.optimizers.RMSprop(learning_rate=0.0001)  # to transform the model for coreML I have to use this optimizer
    # (It also runs slowly on M1 macs but we don't care!)

    model.compile(
        optimizer=adam_optimizer,
        loss='sparse_categorical_crossentropy',     #since I'm not using one-hot encoding I need the sparse version
        metrics=['accuracy']
    )

    model.summary()
    return model


def define_perceptron():
    print("Using PERCEPTRON MODEL")
    model = Sequential()
    model.add(Flatten(input_shape=(WINDOW_SIZE, 6), name='flatten_hidden_Layer'))
    model.add(Dense(64, activation='relu', name='hidden_Layer'))
    model.add(Dense(NUM_CLASSES, activation='softmax', name='output_Layer'))

    adam_optimizer = tf.optimizers.legacy.Adam(learning_rate=0.001)  # performance issues for M1/M2 Macs
    #For all the other cases, it's enough to use...
    # adam_optimizer = tf.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=adam_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def k_fold_cross_validation(data, labels):
    histories = []        # record of training loss values and metrics values at successive epochs,

    # create a k-fold cross-validator
    k_fold = KFold(n_splits=K, shuffle=True)

    # define the model
    model = define_cnn()
    # model = define_perceptron()

    # k-fold cross-validation
    for train, test in k_fold.split(data, labels):
        data_train = data[train]
        data_test = data[test]

        labels_train = labels[train]
        labels_test = labels[test]

        # train the model with training data and labels
        history = model.fit(
            data_train,
            labels_train,
            epochs=EPOCHS,           # an epoch is an iteration over the entire x and y data provided
            batch_size=BATCH_SIZE,   # number of samples per gradient update
            verbose=2,               # one line per epoch
            validation_data=(data_test, labels_test),
            # callbacks=[reduce_lr_acc, early_stopping, mcp_save]
        )
        histories.append(history)
        return model, histories


def validate(model, data, labels):
    loss, accuracy = model.evaluate(data, labels)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

