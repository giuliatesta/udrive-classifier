from convolutional_neural_network import k_fold_cross_validation, validate
from preprocessing import read_csv, data_split, sliding_window

# 1. import data from csv file
data = read_csv("./dataset/sensor_raw_small.csv")

# 2. create sliding windows
windowed_data, windowed_labels= sliding_window(data)

# 3. split data in training and validation set (80/20 ratio)
acc_gyro_train, acc_gyro_test, label_train, label_test = data_split(windowed_data, windowed_labels)

# 4. define the CNN model

# 5. perform k fold cross validation and train the model over the fold
trained_model, histories = k_fold_cross_validation(acc_gyro_train, label_train)

# 6. perform the prediction over the validation test and compute accuracy
validate(trained_model, acc_gyro_test, label_test)