# Deep-Learning-Image-Classification
Deep learning based Image classification using keras and Tensorflow.

In this I have implemented an algorithm that classifies an object in an image. The classification it based on which class that object 
belongs to. The objects could be anything ranging from household objects like a sofa or dining table to humans and animals.
A Convolutional Neural Network (CNN) is created by using python deep learning library keras on tensorflow backend. 

Keras is also used to train the network on the available dataset. Basically there are two ways to initialize the neural network,
either by sequence of layers or as a graph. I used the sequential neural network model by using the sequential class of the keras. 
There are 3 major steps to build a sequential neural network: Convolution, Pooling, Flattening. An object of the sequential() class is
used to add the convolutional layer, pooling layer, and do the flattening. After this, a fully connected layer is created using 
the “Dense” function. This layer is connected to the set of nodes that are obtained after the flattening step.

The nodes act as an input layer to these fully-connected layers. After this, an output layer is added that contains only one node
that will give us a binary output of either the classified image belongs to class A or to class B. This was the final step of
building the CNN model. The CNN is now compiled. 

The data set is then fitted to this CNN after performing image augmentation on the dataset so as to avoid overfitting of nodes from one
layer to another. This is done by keras.preprocessing library.The training of CNN and testing the model is also done by using this library.

The dataset used to train the neural network is available on http://www.superdatascience.com/wp-content/uploads/2017/03/Convolutional-Neural-Networks.zip

