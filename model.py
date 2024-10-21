from nabirds import class_id_to_name 
import tensorflow as tf
from tensorflow.keras import layers, datasets, models #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore
from tensorflow.keras.applications import InceptionV3 #type: ignore

from tensorflow.keras.models import Sequential #type: ignore
import matplotlib.pyplot as plt

def create_model(input_shape=(224,224, 3)):
    num_classes = len(class_id_to_name)

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model = models.Sequential()
    
    model.add(base_model)
    
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.Dropout(0.6))  # Dropout to prevent overfitting
    
    model.add(layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(layers.Dropout(0.6))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


    
