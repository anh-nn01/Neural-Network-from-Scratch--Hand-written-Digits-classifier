# Neural-Network-from-Scratch--Hand-written-Digits-classifier
This is my first Deep Learning project, which is a MNIST hand-written digits classifier. The model is implemented completely from scratch WITHOUT using any prebuilt optimization library like Tensorflow or Pytorch. Tensorflow is imported only to load the MNIST data set. The model reaches an accuracy of ~**99.13%** on the training set and ~**97%** on the test set.

This model consists of 2 hidden layers, each with 400 hidden units and 100 hidden units, respectively. The layers are fully connected, so this is a standard Neural Network. Given the size, this Neural Network is relatively small. The weights and biases are initialized by He et. al (2015) initialization. Each parameters is updated by Adaptive Moment Optimization (Adam) after processing each mini-batch. To prevent potential overfitting, Drop-out regularization is used, with keep probability in the first layer and second layer are 0.4 (40%) and 0.5 (50%), respectively. All hidden units use ReLU activation function, while the output layer use softmax activation function.

Some hyperparameters which I use during training:
+ **α = 0.005**          : learning rate;
+ **α_decay = 0.00001**  : decay rate of learning rate after each epoch;
+ **β1 = 0.8**           : used in momentum gradient descent in adam;
+ **β2 = 0.999**         : used RMSprop in adam;
+ **L = [400, 100, 10]** : number of hidden units in each layers and output layer;
+ **keep_prob = 0.4, 0.5** : keep probability after in first and second hidden layers (do not apply to input and output layer);
+ **mini_batch size** = 2^8.

**Explanation of What behind the scene (How Neural Network learns and How to implement it from scratch)**</br></br>
**To be updated** (I have too many quizzes on school :(()</br></br></br></br>
<img src = "Useful Functions/Neural Network.png"></br></br>
1) First of all, we perform simple forward propagation using the initialized parameters to get the output. To keep everything simple, for now we only consider a single vector representing a single training example. We feed that vector as an input to the neural network to get the output. </br></br>
Suppose a_l_i is the activation of the training data i-th at layer l-th in the neural network. In our convention, a_0_i is the input data for the training example i-th (or a_0_i = x_i). In each layer, the feature vector from layer l-1 is mapped to a new feature vector at layer l. We can perfrom each step of mapping from layer l-1 to layer l using the formula below:  </br></br>
<img src = "Useful Functions/1. Forward Propagation 1.png"></br></br>
Where W_l is the parameter matrix mapping features from layer l-1 to layer l, b_l is the bias vector. W_l and b_l are all parameters that can be optimized to improve the performance of the model. Note that z_l is not the activation of the layer l, but is just the first step. We can then use the result of z_l to compute the activation at layer l a_l by applying nonlinear function on z_l:
</br></br></br></br>

2) Nonlinear function: For all hidden layers (not the ouput layer), we can apply nonlinearity by using the ReLU function (Rectified Linear Unit) like below to get the activations at layer l:</br></br>
<img src = "Useful Functions/2. Forward Propagation 2.png"></br></br>
The function looks something like this:</br></br>
<img src = "Useful Functions/Relu.png"></br></br>
Now we have can get activations at layer l from the activations at the previous layer, l-1. We can repeat this process until we reach the output layer. However, one difference is instead of using ReLU activation function, the output layer uses softmax activation function, which we will discuss later.</br>
**Note**: Intuition of the ReLU activation function: This is my intuition of the ReLU activation function:</br></br>
This function represents something somewhat similar to how human brain works. When doing some tasks, such as looking at an image and recognize whether he or she is your friend or not, there are some sets of neurons that will be activated strongly if it is your friend, inactivated otherwise. The ReLU function maps each input to an output between 0 and +inf, which means the neuron can be either inactivated (zero) or activated weekly (small positive number) or strongly activated (very large positive number). This is the intuition for the ReLU activation function. There are also some advantages of ReLu function over other non-linear function like Sigmoid and Tanh, but we will not discuss specifically what are the advantages here.
</br></br></br></br>

3) So far we have understand how to perform forward propagation for a single training example. However, we get many training example in the data, not just one. How can we perform forward propagation for all of the training examples in the data set?</br></br>
It turns out that it is pretty similar to what we have done in forward propagation for a single training example: We just need to put  feature vectors of each training example together to form a feature matrix, each column of the matrix correspond to the activations of one training example at layer l. Formally, column j of the feature matrix A_l is the activations of training example i-th at layer l-th: </br></br>
<img src = "Useful Functions/3. Activation Matrix.png"></br></br></br></br>

4) Performing Forward Propagation for all Training example:</br>
It is very similar to the formular at section 1). We just change a vector of single training example to a matrix of multiple training example. We just simple expand the number of column while keeping the dimension of each column (number of rows in each column)
<img src = "Useful Functions/4. Forward_Prop.png"></br></br></br></br>
<img src = "Useful Functions/5. Total Loss Function.png"></br></br></br></br>
<img src = "Useful Functions/6. Softmax Activation.png"></br></br></br></br>
<img src = "Useful Functions/8. Backprop1.png"></br></br></br></br>
<img src = "Useful Functions/9. Backprop2.png"></br></br></br></br>
<img src = "Useful Functions/10. Backprop3.png"></br></br></br></br>
<img src = "Useful Functions/11. Backprop4.png"></br></br></br></br>
<img src = "Useful Functions/12. Backprop5.png"></br></br></br></br>
<img src = "Useful Functions/14. Update.png"></br></br></br></br>
<img src = "Useful Functions/15. Update.png"></br></br></br></br>

**Note**: The functions above are written in LaTEX Math Editor online, you can either use the online editor or use my written functions by going to the folder "Useful functions" -> download
