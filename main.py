import os

import plot
from convolutional_neural_network import k_fold_cross_validation, validate
from preprocessing import read_csv, data_split, sliding_window, WINDOW_SIZE

MODEL_PATH = "./model"
MODEL_NAME = f"cnn-{WINDOW_SIZE}"


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# 1. import data from csv file
print("Reading the data...")
data = read_csv("./dataset/final_dataset.csv")
print("Reading done.")

# 2. create sliding windows
print("Preparing the data for training...")
windowed_data, windowed_labels = sliding_window(data)
print("Preparation done.")

# 3. split data in training and validation set (80/20 ratio)
print("Splitting the dataset...")
acc_gyro_train, acc_gyro_test, label_train, label_test = data_split(windowed_data, windowed_labels)
print("Split done.")
# 4. define the CNN model
# 5. perform k fold cross validation and train the model over the fold
print(windowed_labels)
print("Training the model...")
trained_model, histories = k_fold_cross_validation(acc_gyro_train, label_train, cnn=False)
print("Training done.")
# save the model (to be loaded by core_ml_converter)
create_dir(MODEL_PATH)
trained_model.save(f'{MODEL_PATH}/{MODEL_NAME}.keras')
print(f"Saving the model in {MODEL_PATH}")
# 6. perform the prediction over the validation test and compute accuracy
print("Validating the model...")
validate(trained_model, acc_gyro_test, label_test)
print("Validation done.")

# 7. (additional) plots the accuracy values for each history
plot.plot_accuracy(histories, cnn=False)
