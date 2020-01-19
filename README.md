# Neural-Network-from-Scratch--Hand-written-Digits-classifier
This is my first Deep Learning project, which is a MNIST hand-written digits classifier. The model is implemented completely from scratch WITHOUT using any prebuilt optimization library like Tensorflow or Pytorch. Tensorflow is imported only to load the MNIST data set. The model reaches an accuracy of ~97% on the training set and ~99.13% on the test set.

This model consists of 2 hidden layers, each with 400 hidden units and 100 hidden units, respectively. The layers are fully connected, so this is a standard Neural Network. Given the size, this Neural Network is relatively small. The weights and biases are initialized by He et. al (2015) initialization. Each parameters is updated by Adaptive Moment Optimization (Adam) after processing each mini-batch. To prevent potential overfitting, Drop-out regularization is used, with keep probability in each layer is 0.5 (50%). All hidden units use ReLU activation function, while the output layer use softmax activation function.

Some hyperparameters which I use during training:
+ α = 0.005          : learning rate;
+ α_decay = 0.00001  : decay rate of learning rate after each epoch;
+ β1 = 0.8           : used in momentum gradient descent in adam;
+ β2 = 0.999         : used RMSprop in adam;
+ L = [400, 100, 10] : number of hidden units in each layers and output layer;
+ keep_prob = 0.4, 0.5 : keep probability after in first and second hidden layers (do not apply to input and output layer);
+ mini_batch size = 2^8.
