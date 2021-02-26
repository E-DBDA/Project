# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:33:02 2021

@author: Vishakha
"""

"""import splitfolders
input_dataset="dataset"
output_dataset="output"
splitfolders.ratio(input_dataset,output_dataset,seed=42,ratio=(.6,.2,.2))"""

from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import seaborn as sns

RESNET50_POOLING_AVERAGE = 'avg'
img_height,img_width = (224,224)
batch_size = 32

train_data_dir="output/train"
test_data_dir="output/test"
valid_data_dir="output/val"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,validation_split=0.4)

train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_height,img_width),batch_size=32,class_mode='categorical',subset='training')
valid_generator = train_datagen.flow_from_directory(valid_data_dir,target_size=(img_height,img_width),batch_size=32,class_mode='categorical',subset='validation')
test_generator = train_datagen.flow_from_directory(test_data_dir,target_size=(img_height,img_width),batch_size=1,class_mode='categorical',subset='validation')

#x,y = test_generator.next()
#print(x.shape)

base_model = ResNet50(include_top=False,weights="imagenet")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes,activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics = ['accuracy'])
model.fit(train_generator,epochs=15)

#model.save("dataset\savedmode\resnet50.h5")
test_loss, test_acc =model.evaluate(test_generator,verbose = 2)
#print("Test accuracy: ",test_acc)

#model = tf.keras.models.load_model("savedmodel\resnet50_plants.h5")
filenames = test_generator.filenames
nb_samples = len(test_generator)
y_prob = []
y_act = []
test_generator.reset()
for _ in range(nb_samples):
    X_test,Y_test = test_generator.next()
    y_prob.append(model.predict(X_test))
    y_act.append(Y_test)
    
predict_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_prob]
actcual_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_act]

out_df =pd.DataFrame(np.vstack([predict_class,actcual_class]).T,columns=['predict_class','actcual_class'])
conf_matrix = pd.crosstab(out_df['actcual_class'],out_df["predict_class"], rownames=['actual'], colnames=['predicted'])

sns.heatmap(conf_matrix,cmap='Blues',annot=True,fmt='d')
plt.show()

print("Test accuracy : {}".format((np.diagonal(conf_matrix).sum()/conf_matrix.sum().sum()*100)))
