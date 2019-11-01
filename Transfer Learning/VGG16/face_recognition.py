import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

image_size = [224, 224, 3]

vgg = VGG16(input_shape=image_size, weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

folders = glob("datasets/1/*")
x = Flatten()(vgg.output) 
predict = Dense(1, activation='sigmoid')(x)  
model = Model(inputs=vgg.input, outputs=predict)
model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('Datasets',
                                                 target_size=(224, 224),
                                                 class_mode = 'binary')  


r = model.fit_generator(
  training_set,
  validation_data = None,
  epochs=5,
  steps_per_epoch = len(training_set),
)  
 
model.save('facefeatures_new_model.h5')

print("Model trained and saved")