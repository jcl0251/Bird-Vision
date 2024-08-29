from nabirds import class_id_to_name 
import tensorflow as tf
from tensorflow.keras import layers, datasets, models #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore

from tensorflow.keras.models import Sequential #type: ignore
import matplotlib.pyplot as plt

def create_model(input_shape=(224,224, 3)):
    num_classes = len(class_id_to_name)
    model = models.Sequential()
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
    
