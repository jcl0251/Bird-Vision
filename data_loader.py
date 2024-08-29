#Getting started with some imports learned from learnopencv.com
import os
import cv2
import random
import numpy as np #library for python supporting arrays, matrices, and high-level math functions
import matplotlib.pyplot as plt #plotting library for embedding plots into apps with general purpose GUI toolkits
import keras
import PIL
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers #type: ignore
from tensorflow.keras.models import Sequential #type: ignore

def load_data(images_dir, class_id_to_name, img_size=(224, 224)):
    images = []
    labels = []

    for class_id, class_name in class_id_to_name.items():
        class_dir = os.path.join(images_dir, str(class_id)) #class id folder in images folder
        
        if(os.path.exists(class_dir)):
            for img_name in os.listdir(class_dir): #If the class directory that we got from the classes.txt file exists in the images folder, we will go ahead
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = cv2.resize(img, img_size)
                    img = img / 255.0
                    images.append(img)
                    labels.append(class_id)
                    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    return images, labels

def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)

def visualize_sample(images, labels, class_id_to_name, num_samples=25):
    plt.figure(figsize=(10,10))
    for i in range(min(num_samples, len(images))):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_id_to_name[labels[i]])
    plt.show()
    
if __name__ == "__main__":
    from nabirds import images_dir, class_id_to_name
    images, labels = load_data(images_dir, class_id_to_name)
    x_train, x_test, y_train, y_test = split_data(images, labels)
    visualize_sample(x_train, y_train, class_id_to_name)
