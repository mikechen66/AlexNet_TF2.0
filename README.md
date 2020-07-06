# AlexNet_TF2.0 Implementation with tensorflow_datasets

An implementation of AlexNet (Krizhevsky et al.(2010)) written in TensorFlow 2.0.

This code preloads the Oxford_Flowers102 dataset from TensorFlows datasets API. To change which dataset to use
change data_set variable to one from here: https://www.tensorflow.org/datasets/catalog/overview

Author: Henry Powell
Institution: Institute of Neuroscience, Glasgow University, Scotland.
Implementation of AlexNet using Keras with Tensorflow backend. Code will preload the Oxford_Flowers102 dataset.
Learning tracks the model's accuracy, loss, and top 5 error rate. For true comparision of performance to the original
model (Krizhevsky et al. 2010) this implementation will need to be trained on Imagenet2010.

Editor: Mike Chen

It is a good practice for the author to adopt the tensorflow_datasets library. Mike splits the client appliction from 
the AlexNet model for better readibility, accessibiliuty and usability after deleting the function of save_data(). 
Developers need to install the following libraries. 

1.Normally speaking, we only need to install tensorflow_datasets

$ pip install tensorflow_datasets

or 

$ pip3 install tensorflow_datasets

While running the script of main.py, it builds a directory "/home/user/tensorflow_datasets". It has two sub-directories 
including downloads, oxford_flowers102. I assume that developers uses Ubuntu 18.04. 

2.Install tfds-nightly

If it raises NonMatchingChecksumError(resource.url, tmp_path) duruing the runtime, please make the following changes to 
adapt to Oxford which owns the 102flowers data. 

1). Remove the downloaded files

rm -rf ~/tensorflow_datasets/oxford_flowers102/

rm -rf ~/tensorflow_datasets/downloads

2). Install tfds-nightly

$ pip --no-cache-dir install tfds-nightly

or 

$ pip3 --no-cache-dir install tfds-nightly
