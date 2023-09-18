# udrive Classifier

## Convolutional Neural Network

- _Sequential_: create models layer-by-layer by stacking them. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs (like _Functional_).
- _Conv_: since the dataset being accelerometer and gyroscope data, which are _sequential data_, _Conv1D_ is the most appropriate choice.
- _Activation_: uses the _rectifier function_ to allow faster and effective training of deep neural architectures on large and complex datasets. (NB: ReLu is a _ramp_ function)
    The activation function scores pixel values according to some measure of importance. The ReLU activation says that negative values are not important and so sets them to 0. ("_Everything unimportant is equally unimportant._").



### Steps 
1. import data from csv file 
2. create sliding windows 
3. split data in training and validation set (80/20 ratio)
4. define the CNN model 
5. perform k fold cross validation and train the model over the fold 
6. perform the prediction over the validation test and compute accuracy


### Project's specifics 
Use Tensorflow 2 to train a neural network for the binary classification of cats and dogs based on images from this dataset. Images must be transformed from JPG to RGB (or grayscale) pixel values and scaled down. Experiment with different network architectures and training parameters documenting their influence of the final predictive performance. Use 5-fold cross validation to compute your risk estimates. While the training loss can be chosen freely, the reported cross-validated estimates must be computed according to the zero-one loss.
NB: I adapted this project's specific to the problem of driving behaviour classification
### References
- [Driving Behavior Dataset: using Machine learning Predict Driver's Behavior](https://www.kaggle.com/datasets/shashwatwork/driving-behavior-dataset?resource=download)