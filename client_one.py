
"""
Use the command "run_experiment(n, large_data_set=False, generator=False)" to initiate the functional call.
It show pictures, epochs/iterations and final score results but does not save the intermediate big data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import utils
import functools
import numpy as np
import matplotlib.pyplot as plt
from alexnet import AlexNet

# Need to create the fileholder named with Alexnet_Client_Backend
project_path = '/home/mike/Documents/Alexnet_Client_Backend/'
# Need to create the sub-fileholder named data
data_path = '/home/mike/Documents/Alexnet_Client_Backend/data/'

# Set global variables.
epochs = 1
verbose = 1
steps_per_epoch = 100
batch_size = 100
n = 1

# Set the dataset which will be downladed and stored in system memory.
data_set = "oxford_flowers102"

# This list keeps track of the training metrics for each iteration of training if users require to run an experiment,
# if the users want to train the network n times and see how the training differs between the iterations of training.
acc_scores = []
loss_scores = []
top_5_acc_scores = []

# Set up the top 5 error rate metric.
# -top_5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
# -top_5_acc.__name__ = 'top_5_acc'


# Create the Datagenerator class to generate the data in batches to train the network
class DataGenerator(utils.Sequence):

    def __init__(self, image_file_names, labels, batch_size):
        self.image_file_names = image_file_names
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_file_names) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, index):
        batch_image = self.image_file_names[index * self.batch_size: (index+1) * self.batch_size]
        batch_label = self.labels[index * self.batch_size: (index+1) * self.batch_size]

        return np.array(batch_image), np.array(batch_label)


def load_data():
    """
    The Function loads and augment the training, testing and validation data.
    :return: images and labels as numpy arrays (the labels will be one-hot encoded) as well as an info object
    containg information about the loaded dataset.
    """
    # Load the data using TensorFlow datasets API.
    data_train, info = tfds.load(name=data_set, split='test', with_info=True)
    data_test = tfds.load(name=data_set, split='train')
    data_val = tfds.load(name=data_set, split='validation')

    # Ensure that loaded data is of the right type.
    assert isinstance(data_train, tf.data.Dataset)
    assert isinstance(data_test, tf.data.Dataset)
    assert isinstance(data_val, tf.data.Dataset)

    # Print the dataset information.
    print(info)

    return data_train, data_test, data_val, info


def preprocess_data(data_train, data_test, data_val):
    """
    Prerocess the data by applying resizing, augmenting the dataset with rotated and translated versions of each 
    image to prevent the model overfitting.
    :param data_train: tf.data.Dataset object containing the training data
    :param data_test: tf.data.Dataset object containing the test data
    :param data_val: tf.data.Dataset object containing the validation data
    :param augment: Set to True if you want to augment the training data and add it to the dataset for training.
    :return: data_train, data_test, data_val, train_images, train_labels, test_images, test_labels, val_images,
    val_labels - training, test, and validation datasets as tf.data.Dataset objects and individual image, and l
    abel arrays for each.
    """

    # 1. Preprocess train images 

    train_images = []
    train_labels = []

    # We take all the samples in the training set (6149), convert the data type to float32 and resize them. 
    # Since the images in Oxford_Flowers are not preprocessed, we need to resize them all so that the network
    # takes inputs that are all the same size. We will also transform the data to help the network generalize.
    for example in data_train.take(-1):
        # Get images and labels from the Dataset object
        image, label = example['image'], example['label']

        # Resize images and add to dataset
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [227, 227])
        train_images.append(image.numpy())
        train_labels.append(label.numpy())

    
    # 2. Preprocess test images. 
    # Do the same as above but with the test and validation datasets.
    test_images = []
    test_labels = []

    # Do the same as above... 
    for example in data_test.take(-1):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [227, 227])
        test_images.append(image.numpy())
        test_labels.append(label.numpy())

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_labels = utils.to_categorical(test_labels)

    # 3. Preprocess val images. 
    # Do the same as above but with the test and validation datasets.
    val_images = []
    val_labels = []

    # Do the same as above... 
    for example in data_val.take(-1):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [227, 227])
        val_images.append(image.numpy())
        val_labels.append(label.numpy())

    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    val_labels = utils.to_categorical(val_labels)

    return data_train, data_test, data_val, \
           train_images, train_labels, test_images, test_labels, val_images, val_labels


def visualize(data_train, data_test, info):
    """
    The short function visualizes the data set giving 9 samples from each of the training and test datasets
    and their respective labels.
    :param data_train: A tf.data.Dataset object containing the training data
    :param data_test: A tf.data.Dataset object containing the test data
    :param info: dataset.info for getting information about the dataset (number of classes, samples, etc.)
    :return: n/a
    """
    tfds.show_examples(info, data_train)
    tfds.show_examples(info, data_test)


# With the three parameters, it calls the AlexNet model(as a class) in alexnet.py
# -model = AlexNet(train_data, test_data, val_data)

# Give the model structure summary after complete the above-mentioned calling. 
# -model.summary()


def predictions(model, val_images, val_labels, num_examples=1):
    """
    Display some examples of the predicions that the network is making on the testing data.
    :param model: model object
    :param test_images: tf.data.Dataset object containing the training data
    :param test_labels: tf.data.Dataset object containing the testing data
    :return: n/a
    """

    predictions = model.predict(val_images)

    for i in range(num_examples):
        plt.subplot(1, 2, 1)
        # Plot first predicted image
        plt.imshow(val_images[i])

        plt.subplot(1, 2, 2)
        # Plot bar plot of confidence of predictions of possible classes for the first image in the test data
        plt.bar([j for j in range(len(predictions[i]))], predictions[i])
        plt.show()


def run_experiment(n, large_data_set=False, generator=False):
    """
    Run an experiment. One experiment loads the dataset, trains the model, and outputs the evaluation metrics after
    training.
    :param n: Number of experiments to perform
    :param large_data_set: Set to True of you want to save the large dataset to hard disk and use generator for training
    :param generator: Set to True is you want to use a generator to train the network.
    :return: n/a
    """

    for experiments in range(n):

        data_train, data_test, data_val, info = load_data()
        data_train, data_test, data_val,\
        train_images, train_labels, test_images, test_labels,\
        val_images, val_labels = preprocess_data(data_train, data_test, data_val)

        visualize(data_train, data_test, info)


        # Make image, label paris into the tf.data.Dataset, shuffle the data and specify its batch size.
        train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_data = train_data.repeat().shuffle(6149).batch(100)

        test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_data = test_data.repeat().shuffle(1020).batch(batch_size)
            
        val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_data = val_data.batch(batch_size)

        # With the three parameters, the client script calls the AlexNet model(as a class) in alexnet.py
        model = AlexNet(train_data, test_data, val_data)            

        model.fit(train_data,
                  epochs=epochs,
                  validation_data=val_data,
                  verbose=verbose,
                  steps_per_epoch=steps_per_epoch)

        # Give the model structure summary after complete the above-mentioned calling. 
        model.summary()

        predictions(model, test_images, test_labels, num_examples=5)

        # Evaluate the model
        loss, accuracy, top_5 = model.evaluate(test_data, verbose=verbose, steps=5)

        # Append the metrics to the scores lists in case you are performing an experiment which involves comparing
        # training over many iterations.
        loss_scores.append(loss)
        acc_scores.append(accuracy)
        top_5_acc_scores.append(top_5)

        # return AlexNet(as model--hidden the return statement)

        # Print the mean, std, min, and max of the validation accuracy scores from your experiment.
        print(acc_scores)
        print('Mean_accuracy={}'.format(np.mean(acc_scores)), 'STD_accuracy={}'.format(np.std(acc_scores)))
        print('Min_accuracy={}'.format(np.min(acc_scores)), 'Max_accuracy={}'.format(np.max(acc_scores)))


run_experiment(n, large_data_set=False, generator=False)
