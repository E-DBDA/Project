# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:10:16 2021
@author: Hitakshi

EfficientNetB0 Transfer Learning implementation
model is giving 94% accuracy
data is split in 60-40 pattern
total 1835 segmented Images 

"""
#Need to install efficientnet using this command 
#!pip install -U efficientnet

import efficientnet.keras as efn
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import Model 
from tensorflow import keras


import splitfolders
input_dataset="C:/Users/Hitakshi/Downloads/trail/Input"
output_dataset="C:/Users/Hitakshi/Downloads/trail/Output"
splitfolders.ratio(input_dataset,output_dataset,seed=42,ratio=(.6,.4))


train_data="C:/Users/Hitakshi/Downloads/trail/Output/train"
test_data="C:/Users/Hitakshi/Downloads/trail/Output/val"


# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255., 
                                   rotation_range = 40, 
                                   width_shift_range = 0.2, 
                                   height_shift_range = 0.2, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_data, batch_size = 35, 
                                                    class_mode = 'categorical', 
                                                    target_size = (224, 224))

test_generator = test_datagen.flow_from_directory( test_data, batch_size = 35, 
                                                        class_mode = 'categorical', 
                                                        target_size = (224, 224))

model = efn.EfficientNetB0(input_shape = (224, 224, 3), 
                                include_top = False,
                                weights = 'imagenet')

for layer in model.layers:
    layer.trainable = False
    
    
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model_final = Model(inputs = model.input, outputs = predictions)

model_final.compile(keras.optimizers.Adam(lr=0.0001, decay=1e-6),
                    loss='categorical_crossentropy',metrics=['accuracy'])

Run = model_final.fit_generator(train_generator, 
                                validation_data = test_generator,
                                epochs = 10)

test_loss, test_acc =model_final.evaluate(test_generator,verbose = 2)
print("Test accuracy: ",test_acc)


from tensorflow.keras.models import load_model

model.save('C:/Users/Hitakshi/Downloads/trail/Effectiveb0.h5')
model=load_model('C:/Users/Hitakshi/Downloads/trail/Effectiveb0.h5')