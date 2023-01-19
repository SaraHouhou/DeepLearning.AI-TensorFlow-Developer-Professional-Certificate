
import os
import numpy as np
from keras.preprocessing import image




def predict(VALIDATION_DIR, MODEL, IMAGE_SIZE):
    images = os.listdir(VALIDATION_DIR)
    for i in images:
        print()
# predicting images
        path = VALIDATION_DIR + i       
        img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
        images = np.vstack([x])
        classes = MODEL.predict(images, batch_size=10)
        print(path)
        print(classes)