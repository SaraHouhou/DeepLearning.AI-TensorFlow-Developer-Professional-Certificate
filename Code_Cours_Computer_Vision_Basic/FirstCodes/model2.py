
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






num_classes=2
image_size=128
batch_size_training=20
batch_size_validation=20

#BATCH_SIZE (integer): Integer number of batches (amount of images trained per loop).
#INPUT_SIZE (tensor: (int, int)): Tensor containing the size of the input image, or 

# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR, BATCH_SIZE, IMAGE_SIZE, CLASS_MODE):
  """
  Creates the training and validation data generators
  
  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  train_datagen = ImageDataGenerator(rescale=1./255.0,
                                    #  rotation_range=40,
                                    #  width_shift_range=0.2,
                                    #  height_shift_range=0.2,
                                    #  shear_range=0.2,
                                    #  zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    #  fill_mode='nearest'
                                    )

  # Pass in the appropiate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=BATCH_SIZE,
                                                      class_mode=CLASS_MODE,
                                                      target_size=(IMAGE_SIZE, IMAGE_SIZE))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator( rescale = 1.0/255.)

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=BATCH_SIZE,
                                                                class_mode=CLASS_MODE,
                                                                target_size=(IMAGE_SIZE,IMAGE_SIZE))
  ### END CODE HERE
  return train_generator, validation_generator

def create_model(NUM_CLASSES):
    model = tf.keras.models.Sequential([
        VGG16(include_top=False, pooling='avg', weights="imagenet"),
        Dense(NUM_CLASSES, activation='softmax') 
        ])

    return model



base_dir = 'simples/'

print("Contents of base directory:")
print(os.listdir(base_dir))

train_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')

print("\nContents of train directory:")
print(os.listdir(train_dir))

print("\nContents of validation directory:")
print(os.listdir(validation_dir))



train_generator, validation_generator=train_val_generators(train_dir,validation_dir, BATCH_SIZE=20, IMAGE_SIZE=224, CLASS_MODE='categorical')
model=create_model(NUM_CLASSES=2)
model.summary()
model.layers[0].layers

# take the pretrained on the imagenet and train it in the input and the output of our images
model.layers[0].trainable=False
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
steps_per_epoch_train=len(train_generator)
steps_per_epoch_valid=len(validation_generator)


history = model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=2,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=steps_per_epoch_valid)