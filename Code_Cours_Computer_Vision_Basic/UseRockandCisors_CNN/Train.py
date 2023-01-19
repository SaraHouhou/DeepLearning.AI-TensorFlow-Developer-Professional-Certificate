
# Dataset from https://laurencemoroney.com/datasets.html

# Images have all been generated using CGI techniques as an experiment in determining if a CGI-based dataset can be used for classification against real images.
import LoadData
import designPlot
import model
import predict
import tensorflow as tf
# load Data

base_dir="Dataset"

train_dir, test_dir, validation_dir= LoadData.load(base_dir)

LoadData.Show_NB_Data(train_dir, test_dir, validation_dir)
# Data preprocessing

train_generator, test_generator=LoadData.train_val_generators(train_dir, test_dir, 150, 10, 'categorical')


# configure the network
IMAGE_SIZE=150
NBClasses=3
network = model.create_model(IMAGE_SIZE,NBClasses)

# Print the model summary
network.summary()

# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.96 & logs.get('loss')>0.91):
      print("\nReached 96% accuracy so cancelling training!")
      self.model.stop_training = True

# Set the training parameters
callbacks = myCallback()
network.compile(loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
    reduction='auto'
), optimizer='adam', metrics=['accuracy'])

# Train the model

history = network.fit(train_generator, epochs=500,  validation_data = test_generator, verbose = 1)

# design results

designPlot.diagrams(history=history)


# Model Prediction
#predict(validation_dir, network, 150 )





