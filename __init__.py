#Getting started with some imports learned from learnopencv.com
import os
import random
import numpy as np #library for python supporting arrays, matrices, and high-level math functions
import matplotlib.pyplot as plt #plotting library for embedding plots into apps with general purpose GUI toolkits
import keras
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers #type: ignore
from tensorflow.keras.models import Sequential #type: ignore

nabirds_dir = os.path.dirname(__file__) #Find the location of the script we're in
images_dir = os.path.join(nabirds_dir, 'images') #Find the images folder which should be in the current directory
classes_file = os.path.join(nabirds_dir, 'classes.txt') #Finds the classes file which has all of our IDs and bird names

class_id_to_name = {}

with open(classes_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            class_id, class_name = line.split(' ', 1) #split by space because format is ID then name
            class_id_to_name[int(class_id)] = class_name #dictionary pairing class_id to name

def get_class_name_by_id(class_id):
    return class_id_to_name.get(class_id, "Unknown")

if __name__ == "__main__":
    for class_id, class_name in class_id_to_name.items():
        print(f"ID: {class_id}, Name: {class_name}")

