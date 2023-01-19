""" 
# we use 4 convolution layers with 64-64-128-128 filters then append a Dropout layer to avoid overfitting and some Dense layers for the classification. 
# The output layer would be a 3-neuron dense layer activated by Softmax. """
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

import tensorflow as tf
from keras.applications.vgg16 import VGG16

def create_model_with_four_cnn(IMAGE_SIZE, NBCLASSES):
    model =tf.keras.models.Sequential([
    # Note the input shape is the desired size of the imagesize with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(NBCLASSES, activation='softmax')
    ])

    return model

def create_model_with_two_cnn(IMAGE_SIZE, CLASSESN):
   
  ### START CODE HERE       

  # Define the model
  # Use no more than 2 Conv2D and 2 MaxPooling2D
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
   # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(CLASSESN, activation='softmax')])
  


  ### END CODE HERE       
  
  return model





   # a tester https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy