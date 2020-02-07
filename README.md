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

1) First of all, we perform simple forward propagation using the initialized parameters to get the output. To keep everything simple, for now we only consider a single vector representing a single training example. We feed that vector as an input to the neural network to get the output. </br>
Suppose a_l_i is the activation of the training data i-th at layer l-th in the neural network. In our convention, a_0_i is the input data for the training example i-th (or a_0_i = x_i). In each layer, the feature vector from layer l-1 is mapped to a new feature vector at layer l. We can perfrom each step of mapping from layer l-1 to layer l using the formula below:  </br></br>
<img src = "Useful Functions/1. Forward Propagation 1.png"></br>
Where W_l is the parameter matrix mapping features from layer l-1 to layer l, b_l is the bias vector. W_l and b_l are all parameters that can be optimized to improve the performance of the model. Note that z_l is not the activation of the layer l, but is just the first step. We can then use the result of z_l to compute the activation at layer l a_l by applying nonlinear function on z_l:
</br></br></br></br>
<img src = "Useful Functions/2. Forward Propagation 2.png"></br></br></br></br>
<img src = "Useful Functions/3. Activation Matrix.png"></br></br></br></br>
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
