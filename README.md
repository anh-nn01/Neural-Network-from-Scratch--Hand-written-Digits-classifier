# Neural-Network-from-Scratch--Hand-written-Digits-classifier
This is my first Deep Learning project, which is a MNIST hand-written digits classifier. The model is implemented completely from scratch WITHOUT using any prebuilt optimization library like Tensorflow or Pytorch. Tensorflow is imported only to load the MNIST data set. The model reaches an accuracy of ~**99.13%** on the training set and ~**97%** on the test set.

Model Description:
===============
This model consists of 2 hidden layers, each with 400 hidden units and 100 hidden units, respectively. The layers are fully connected, so this is a standard Neural Network. Given the size, this Neural Network is relatively small. The weights and biases are initialized by He et. al (2015) initialization. Each parameters is updated by Adaptive Moment Optimization (Adam) after processing each mini-batch. To prevent potential overfitting, Drop-out regularization is used, with keep probability in the first layer and second layer are 0.4 (40%) and 0.5 (50%), respectively. All hidden units use ReLU activation function, while the output layer use softmax activation function.

Hyperparameters:
===============
+ **α = 0.005**          : learning rate;
+ **α_decay = 0.00001**  : decay rate of learning rate after each epoch;
+ **β1 = 0.8**           : used in momentum gradient descent in adam;
+ **β2 = 0.999**         : used RMSprop in adam;
+ **L = [400, 100, 10]** : number of hidden units in each layers and output layer;
+ **keep_prob = 0.4, 0.5** : keep probability after in first and second hidden layers (do not apply to input and output layer);
+ **mini_batch size** = 2^8.

**How Neural Network learns and How to implement it from scratch**
===============
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
It is very similar to the formula at section 1) and 2). We just change a vector of single training example to a matrix of multiple training example. We simply expand the number of column while keeping the dimension of each column (number of rows in each column)</br></br>
<img src = "Useful Functions/4. Forward_Prop.png"></br></br></br></br>
5) Unlike the activations in hidden layers, the output layer use softmax function as its non-linear function instead of ReLu. This function produces outputs between 0 and 1, with probability closer to 1 correspond to higher chance of a training example belongs to one specific class, closer to 0 otherwise. The total probabilities that a training example belong to one specific class is equal to 1, and we will take class with highest probability to assign the training example to such class. This is why the function is called "softmax".</br></br>
<img src = "Useful Functions/6. Softmax Activation.png"></br></br></br></br>

6) After performing forward propagation on the entire training set, we get the outputs for each training example, which is the probabilites that a training example belongs to a class. The outputs produced by the current model might have very high error compared to the ground truth, and we need to define a function to measure the error and to minimize such error so that the model achieve the optimal performance. Such function is called the Cost function, as defined below: </br></br>
<img src = "Useful Functions/5. Total Loss Function.png"></br></br>
Some might wonder why the cost function looks like this. Here a brief explanation:</br></br>
We want the error to be large when the model produce incorrect output and to be low otherwise. For example, if the neural network sees a man and it classifies him as a car, then we want the error to be high (indicating the model is not doing as well as expectation), but if if correctly classifies a man as a man, the error should be low.</br></br>
In a formal definition, if y_hat is the same as y (correctly classify), then the larger the y_hat, the lower the error (better confidence in correct classification). However, if y_hat is different from y (wrong classification), then the larger the y_hat, the higher the error (higher confidence in incorrect prediction). We can represent this idea using the log function:</br></br>
</br> -log x: </br></br>
<img src = "Useful Functions/-log(x).png"></br></br>
</br> -log(1-x): </br></br>
<img src = "Useful Functions/-log(1-x).png"></br></br>
Observe the function -log(y_hat), the closer y_hat to 1, the lower the error, and closer y_hat to 0, the higher the error. This happens when the model is producing correct prediction. Similarly, in the function -log(1-y_hat), the closer y_hat to 1, the higher the error, and closer y_hat to 0, the lower the error. This happens when the model is producing incorrect prediction.</br></br>

By having a cost function as an objective for the optimization, we can apply various optimization technique on the function to minimize the cost, thus improving the model's performance.

</br></br></br></br>
7. Now we have the optimization objective: the Cost function. Recall that the parameters we need to optimize are W_l and b_l, with l = 1, 2,..., L, where L is the total number of layers in the neural network. To optimize such parameters, the most basic way to do is to apply gradient descent on the cost function, and then update each parameteres using such gradients. The formulas for computing gradient descent are the formulas below, all of which can be easily proven using matrix calculus:</br></br>
<img src = "Useful Functions/8. Backprop1.png"></br></br></br></br>
<img src = "Useful Functions/9. Backprop2.png"></br></br></br></br>
<img src = "Useful Functions/10. Backprop3.png"></br></br></br></br>
<img src = "Useful Functions/11. Backprop4.png"></br></br></br></br>
<img src = "Useful Functions/12. Backprop5.png"></br></br></br></br>

8. After finding the correct gradients for each parameters, the final step is to update each parameters: </br></br>
<img src = "Useful Functions/Update1.png"></br></br></br></br>
<img src = "Useful Functions/Update2.png"></br></br></br>
