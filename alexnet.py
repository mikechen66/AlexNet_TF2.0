# AlexNet Model 

"""
AlexNet, Krizhevsky, Alex, Ilya Sutskever and Geoffrey E. Hinton, 2012
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
Michael Guerzhoy and Davi Frossard, 2016
AlexNet implementation in TensorFlow, with weights Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

Editor: Mike Chen
The script is the realization of object oriented style of AlexNet. It is the AlexNet model called by the 
client application. Its construction method of AlexNet includes three parameters, including self, train_data, 
test_data, val_data, generator=False
"""


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import keras
from keras import utils
import numpy as np
import functools


# Set up the top 5 error rate metric.
top_5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top_5_acc.__name__ = 'top_5_acc'

# Define the AlexNet model in the sequential superclass. 
class AlexNet(Sequential):

    def __init__(self, train_data, test_data, val_data, generator=False):

        """
        Build, compile, fit, and evaluate the AlexNet model using Keras.
        :param train_data: a tf.data.Dataset object containing (image, label) tuples of training data.
        :param test_data: a tf.data.Dataset object containing (image, label) tuples of test data.
        :param val_data: a tf.data.Dataset object containing (image, label) tuples of validation data.
        :param generator: Set to true if using a generator to train the network.
        :return: trained model objects.
        """

        super().__init__()

        # 1st Convolutional Layer: (227-11+2*0)/4 + 1 = 55
        self.add(Conv2D(input_shape=(227,227,3), kernel_size=(11, 11), strides=(4,4), padding="valid",
                        filters=96, activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
        self.add(BatchNormalization())

        # 2nd Convolutional Layer: (27-5+2*2)/1 + 1 = 27
        self.add(Conv2D(kernel_size=(5, 5), strides=(1, 1), padding="same",
                        filters=256, activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
        self.add(BatchNormalization())

        # 3rd Convolutional Layer: (13-3+ 2*1)/1 + 1 = 13
        self.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same",
                        filters=384, activation='relu'))
        self.add(BatchNormalization())

        # 4th Convolutional Layer: (13-3+2*1) + 1 = 13
        self.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same",
                        filters=384, activation='relu'))
        self.add(BatchNormalization())

        # 5th Convolutional Layer: (13-3+2*1) + 1 = 13
        self.add(Conv2D(kernel_size=(3,3), strides=(1,1), padding="same",
                        filters=256, activation='relu'))
        self.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
        self.add(BatchNormalization())

        # Flatten 256 x 6 x 6 = 9216 as one dimention vector and pass it to the 6th Fully
        # Connected layer
        self.add(Flatten())

        # 6th layer: fully connected layer with 4096 neurons with 50% dropout and batch normalization.
        self.add(Dense(units=4096, activation='relu'))
        self.add(Dropout(rate=0.5))
        self.add(BatchNormalization())

        # 7th layer: fully connected layer with 4096 neurons with 50% dropout and batch normalization.
        self.add(Dense(units=4096, activation='relu'))
        self.add(Dropout(rate=0.5))
        self.add(BatchNormalization())

        # 8th Layer: fully connected layer with 1000 neurons with 50% dropout and batch normalization.
        self.add(Dense(units=1000, activation='relu'))
        self.add(BatchNormalization())

        # Output layer: softmax function of 102 classes of the dataset. This integer should be changed to match
        # the number of classes in your dataset if you change from Oxford_Flowers.
        self.add(Dense(units=102, activation='softmax'))

        # Compile the model using Adam optimizer and categorical_crossentropy loss function.
        self.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['acc', top_5_acc])
