
#Files 
import os
import pathlib

# Visualization
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Preprocessing & Evaluation
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
In this project we will be using Tensorflow's CNN Architecture, 
to create a multilayer classifier that's capable of reading and predicting the differences in the presented class of animals, using the patterns learnt by the neurons in the Neural networksfrom our input trainng dataset.

The Dataset was downloaded from https://www.kaggle.com/code/husnakhan/animal-faces/data.

Dataset Class_names are as follows: `[Cat, Dog, Wild]`


"""


train_dir = "/content/drive/MyDrive/afhq/train"
val_dir = "/content/drive/MyDrive/afhq/val"


"""Our next step is to prepare the dataset and have it ready for the model. There are a few steps we will do to carry out this process, these are:

* ImageDataGenerator - For our algorithm to be able to read our file.

* Augmentation(train Dataset) - Aim of giving the model a better chance of undestanding the dataset, and possibly produce a more reliable pattern that can help return a better prediction accuracy.

* Normalization - Scaling the dataset (between 0 and 1), easier for the model to read an also faster.


"""

# PREPROCESSING



# During ImageDataGenerator process we will carry out image augmentation on train and test dataset to manipulate the appearance of our dataset.

train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             shear_range=0.2, 
                                             zoom_range=0.2,
                                             width_shift_range=0.2, 
                                             height_shift_range=0.3, 
                                             horizontal_flip=True


test_datagen = ImageDataGenerator(rescale=1/255.)


train_ds_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                target_size=(224,224),
                                                                batch_size=32,
                                                                class_mode="categorical",
                                                                )


val_ds = test_datagen.flow_from_directory(val_dir,
                                          target_size=(224,224),
                                          batch_size=32,
                                          class_mode="categorical",
                                          )


print(train_ds_augmented.image_shape, train_ds_augmented.num_classes)


augmented_images, augmented_labels = train_ds_augmented.next()

random_number = random.randint(0, 32)

plt.imshow(augmented_images[random_number])
plt.title("Augmented image")
plt.axis(False);



"""
In this section I will build a model using a convolutional Neural Network architecture. The dataset has three classes,
this means we will build a model that has 3 output layers to accommodate this,this is known as ` Multilayer classification`


We will also implement two callback functions (Lr_schedule, EarlyStopping) to potentially aid us during training.


"""


model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(224, 224, 3),
                           filters=10,
                           kernel_size=3,
                           padding="valid",
                           activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
     tf.keras.layers.Dense(128, activation="relu"),
     tf.keras.layers.MaxPooling2D(),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(3, activation="softmax")
])


model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"]
                )



def lr_schedule(epoch, lr):
  
  "multiplies the learning rate by exponent of -1, after 5 epochs"
  
  if epoch <= 5:
    return lr
  return lr * tf.math.exp(-0.1)


lr_rate = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)


history_1 = model_1.fit(train_ds_augmented,
                      epochs=10,
                      batch_size=32,
                      steps_per_epoch=len(train_ds_augmented),
                      validation_data= val_ds,
                      validation_steps= len(val_ds),
                      callbacks=[lr_rate, early_stop]
                      )



print(model_1.summary())

"""
Lets plot the model's history so that we can visualize the training pattern of model.

We noticed the training dataset has a smooth curve, we also noticed that the validation dataset was showed hint of potential overfitting,
however after 6 epochs this improved and  maintained an promising validation pattern, perhaps it could be the effect of the lr_schedule callback.


"""

loss = history_1.history["loss"]
val_loss = history_1.history["val_loss"]

accuracy = history_1.history["accuracy"]
val_accuracy = history_1.history["val_accuracy"]


plt.plot(loss, label="train_ds_loss")
plt.plot( val_loss, label="val_loss")
plt.title("loss")
plt.xlabel("epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(accuracy, label="train_ds_accuracy")
plt.plot(val_accuracy, label="val_accuracy")
plt.title("accuracy")
plt.xlabel("epochs")
plt.legend()
plt.show();


model_1.save("saved_animal_model_1.h5")
