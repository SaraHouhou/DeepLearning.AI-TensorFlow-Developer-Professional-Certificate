# -*- coding: utf-8 -*-
#Created on Tue Sep 27 16:27:06 2022

#@author: shouhou
#"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.optimizers import RMSprop,ADAM
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, UpSampling2D, Activation

base_dir = 'Data/'

print("Contents of base directory:")
print(os.listdir(base_dir))

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

print("\nContents of train directory:")
print(os.listdir(train_dir))

print("\nContents of validation directory:")
print(os.listdir(validation_dir))

# Directory with training call/menu pictures
train_call_dir = os.path.join(train_dir, 'call')
train_menu_dir = os.path.join(train_dir, 'menu')

# Directory with validation call/menu pictures
validation_call_dir = os.path.join(validation_dir, 'call')
validation_menu_dir = os.path.join(validation_dir, 'menu')

train_call_fnames = os.listdir( train_call_dir )
train_menu_fnames = os.listdir( train_menu_dir )

print(train_call_fnames[:10])
print(train_menu_fnames[:10])



#let's find out the total number of call and menu images in the train and validation directories:
    
    
print('total training call images :', len(os.listdir(train_call_dir ) ))
print('total validation call images :', len(os.listdir(validation_call_dir ) ))

print(' total training menu images :', len(os.listdir(train_menu_dir) ))
print('total validation menu images :', len(os.listdir(validation_menu_dir ) ))

#take a look at a few pictures to get a better sense of what the call and menu datasets look like. First, configure the matplotlib parameters:
# Parameters for our graph; we'll output images in a 4x4 configuration
# nrows = 4
# ncols = 4

# pic_index = 0 # Index for iterating over images

# #Display a batch of 8 call and 8 menu pictures. You can re-run the cell to see a fresh batch each time:
    
    
#     # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols*4, nrows*4)

# pic_index+=8

# next_call_pix = [os.path.join(train_call_dir, fname) 
#                 for fname in train_call_fnames[ pic_index-8:pic_index] 
#                ]

# next_menu_pix = [os.path.join(train_menu_dir, fname) 
#                 for fname in train_menu_fnames[ pic_index-8:pic_index]
#                ]

# for i, img_path in enumerate(next_call_pix+next_menu_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)

#   img = mpimg.imread(img_path)
#   #print(img.shape)

# plt.show()



#To train a neural network to handle the images, you'll need them to be in a uniform size

#We defined a Sequential layer as before, adding some convolutional layers first. 
#Note the input_shape parameter this time. Here is where we put the 150x150 size and 3 for the color depth 
#because WE have colored images. 
#we then add a couple of convolutional layers and flatten the final result to feed into the densely connected layers.

# model = tf.keras.models.Sequential([
#     # Note the input shape is the desired size of the image 150x150 with 3 bytes color
#     tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2), 
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
#     tf.keras.layers.MaxPooling2D(2,2),
#     # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(), 
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(512, activation='relu'), 
#     # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
#     tf.keras.layers.Dense(1, activation='sigmoid')  
# ])



model = tf.keras.models.Sequential([
      tf.keras.applications.VGG16(include_top=False, input_shape=(150, 150, 3)),
       # Flatten the results to feed into a DNN
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024, activation='relu'), 
      tf.keras.layers.Dense(512, activation='relu'),
      # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
      tf.keras.layers.Dense(1, activation='sigmoid') ])



model.summary()


# preprocessing 
#Next step is to set up the data generators that will read pictures in the source folders, convert them to float32 tensors, and feed them (with their labels) to the model. 

# All images will be rescaled by 1./255.

#As you may already know, data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network (i.e. It is uncommon to feed raw pixels into a ConvNet.) In this case, you will preprocess the images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range).

#In Keras, this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory).
#
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))     
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

# The loss and accuracy are great indicators of progress in training. 
# loss measures the current model prediction against the known labels, calculating the result. 
# accuracy, on the other hand, is the portion of correct guesses.
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)

# network.compile(optimizer=RMSprop(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics = ['accuracy'])

model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics = ['accuracy'])

#Training
history = model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
            verbose=2
            )

print(type(history))

# Plot courbe d'apprentissage

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history[:, 0], label='train loss')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(history[:, 1], label='train acc')
# plt.legend()
# plt.show()

#  plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss, label='train loss')
#     plt.plot(test_loss, label='test loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc, label='train acc')
#     plt.plot(test_acc, label='test acc')
#     plt.legend()
#     plt.show()


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
plt.show()

#remarque : As you can see, the model is overfitting like it's getting out of fashion. 
# The training accuracy (in blue) gets close to 100% while the validation accuracy (in orange) stalls as 70%. 
# The validation loss reaches its minimum after only five epochs.

# Since you have a relatively small number of training examples (2000), overfitting should be the number one concern. Overfitting happens when a model exposed to too few examples learns patterns that do not generalize to new data, i.e. when the model starts using irrelevant features for making predictions. For instance, if you, as a human, only see three images of people who are lumberjacks, and three images of people who are sailors, and among them the only person wearing a cap is a lumberjack, you might start thinking that wearing a cap is a sign of being a lumberjack as opposed to a sailor. You would then make a pretty lousy lumberjack/sailor classifier.

# Overfitting is the central problem in machine learning: given that you are fitting the parameters of our model to a given dataset, how can you make sure that the representations learned by the model will be applicable to data it has never seen before? How do you avoid learning things that are specific to the training data?