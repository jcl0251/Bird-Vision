import tensorflow as tf
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from nabirds.model import create_model
from nabirds.data_loader import load_data, split_data
from nabirds import images_dir, class_id_to_name
import matplotlib.pyplot as plt #plotting library for embedding plots into apps with general purpose GUI toolkits
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


images, labels = load_data(images_dir, class_id_to_name)
x_train, x_test, y_train, y_test = split_data(images, labels)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

#y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(class_id_to_name))
#y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(class_id_to_name))


#Creates the model
model = create_model(input_shape=(224,224,3))

#Compiles the model
#KEEP LEARNING RATE AT THIS LEVEL
model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

data_augmentation = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
data_augmentation.fit(x_train)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_custom.keras', save_best_only=True)

#View layers
model.summary()

#Train the model
history = model.fit(
    data_augmentation.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=50,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('final_model.keras')


