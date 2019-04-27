# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
#import tensorflow as tf
#from keras.models import model_from_json

#from IPython.display import display
#from PIL import image
#import tensorflow

# In line 2, we’ve imported Sequential from keras.models, to initialise our neural network model as a sequential network.
# There are two basic ways of initialising a neural network, either by a sequence of layers or as a graph.

#In line 3, we’ve imported Conv2D from keras.layers, this is to perform the convolution operation i.e the first step of
# a CNN, on the training images. Since we are working on images here, which a basically 2 Dimensional arrays, we’re using
# Convolution 2-D, you may have to use Convolution 3-D while dealing with videos, where the third dimension will be time.

#In line 4, we’ve imported MaxPooling2D from keras.layers, which is used for pooling operation, that is the step — 2 in
# the process of building a cnn. For building this particular neural network, we are using a Maxpooling function, there
# exist different types of pooling operations like Min Pooling, Mean Pooling, etc. Here in MaxPooling we need the maximum
# value pixel from the respective region of interest.

#In line 5, we’ve imported Flatten from keras.layers, which is used for Flattening. Flattening is the process of
# converting all the resultant 2 dimensional arrays into a single long continuous linear vector.

#In line 6, we’ve imported Dense from keras.layers, which is used to perform the full connection of the neural network,
# which is the step 4 in the process of building a CNN.

#Now, we will create an object of the sequential class below:
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Let’s break down the above code function by function. We took the object which already has an idea of how our neural
# network is going to be(Sequential), then we added a convolution layer by using the “Conv2D” function. The Conv2D
# function is taking 4 arguments, the first is the number of filters i.e 32 here, the second argument is the shape each
# filter is going to be i.e 3x3 here, the third is the input shape and the type of image(RGB or Black and White)of each
# image i.e the input image our CNN is going to be taking is of a 64x64 resolution and “3” stands for RGB, which is a
# colour img, the fourth argument is the activation function we want to use, here ‘relu’ stands for a rectifier function.

#Now, we need to perform pooling operation on the resultant feature maps we get after the convolution operation is done
# on an image. The primary aim of a pooling operation is to reduce the size of the images as much as possible.
# The key thing to understand here is that we are trying to reduce the total number of nodes for the upcoming layers.

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# We start by taking our classifier object and add the pooling layer. We take a 2x2 matrix we’ll have minimum pixel loss
# and get a precise region where the feature are located.  We just reduced the complexity of the model without reducing
# it’s performance.

# Convert all the pooled images into a continuous vector through Flattening. What we are basically doing here is taking
# the 2-D array, i.e pooled image pixels and converting them to a one dimensional single vector.

classifier.add(Flatten())

# In this step we need to create a fully connected layer, and to this layer we are going to connect the set of nodes we
# got after the flattening step, these nodes will act as an input layer to these fully-connected layers. As this layer
# will be present between the input layer and output layer, we can refer to it a hidden layer.

classifier.add(Dense(units = 128, activation = 'relu'))

# Dense is the function to add a fully connected layer, ‘units’ is where we define the number of nodes that should be
# present in this hidden layer, these units value will be always between the number of input nodes and the output nodes
# but the art of choosing the most optimal number of nodes can be achieved only through experimental tries.

# Intialize the output layer. This layer should contain only one node as it is binary classification.
# This single node will give us a binary output of either a Cat or Dog.

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# It’s time to fit our CNN to the image dataset that you’ve downloaded.But before we do that, we are going to pre-process
# the images to prevent over-fitting. Overfitting is when you get a great training accuracy and very poor test accuracy due
# to overfitting of nodes from one layer to another.
#
# So before we fit our images to the neural network, we need to perform some image augmentations on them, which is
# basically synthesising the training data. We are going to do this using keras.preprocessing library for doing the
# synthesising part as well as to prepare the training set as well as the test test set of images that are present in a
# properly structured directories, where the directory’s name is take as the label of all the images present in it.
# For example : All the images inside the ‘cats’ named folder will be considered as cats by keras.

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r'C:\Users\vedan\Desktop\Computer Vision\Final Project\training_set\training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory(r'C:\Users\vedan\Desktop\Computer Vision\Final Project\test_set\test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

# Now lets fit the data to our model

classifier.fit_generator(training_set,
steps_per_epoch = 1589,
epochs = 10,
validation_data = test_set,
validation_steps = 2000)
# In the above code, ‘steps_per_epoch’ holds the number of training images, i.e the number of images the training_set folder contains.
# And ‘epochs’, A single epoch is a single step in training a neural network; in other words when a neural network is
# trained on every training samples only in one pass we say that one epoch is finished. So training process should consist
# more than one epochs.In this case we have defined 25 epochs.

# One way of saving the trained model: using tensorflow
# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, 'my_test_model')

# Another way of saving trained model: We save using this way
# Save the model on disk
# serialize model to JSON
classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

# TESTING our trained model:
# The test_image holds the image that needs to be tested on the CNN. Once we have the test image, we will prepare the
# image to be sent into the model by converting its resolution to 64x64 as the model only excepts that resolution.
# Then we are using predict() method on our classifier object to get the prediction. As the prediction will be in a
# binary form, we will be receiving either a 1 or 0, which will represent a dog or a cat respectively.
test_image = image.load_img(r'C:\Users\vedan\Desktop\Computer Vision\Final Project\test_set\test_set\cats\cat.4023.jpg',
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print('Object=', prediction)