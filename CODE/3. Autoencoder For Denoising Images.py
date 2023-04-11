# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:31:39 2023

@author: noopa
"""
import matplotlib.pyplot as plt
import random
%matplotlib inline

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np

# input layer
input_img = Input(shape=(784,))

# autoencoder
encoding_dim = 32  
encoded = Dense(encoding_dim, activation='relu')(input_img)
encoded_input = Input(shape=(encoding_dim,))
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# get MNIST images, clean and with noise
def get_mnist(noise_factor=0.5):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
  x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

  x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
  x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

  x_train_noisy = np.clip(x_train_noisy, 0., 1.)
  x_test_noisy = np.clip(x_test_noisy, 0., 1.)
  
  return x_train, x_test, x_train_noisy, x_test_noisy, y_train, y_test

x_train, x_test, x_train_noisy, x_test_noisy, y_train, y_test = get_mnist()


# plot n random digits
# use labels to specify which digits to plot
def plot_mnist(x, y, n=10, randomly=False, labels=[]):
  plt.figure(figsize=(20, 2))
  if len(labels)>0:
    x = x[np.isin(y, labels)]
  for i in range(1,n,1):
      ax = plt.subplot(1, n, i)
      if randomly:
        j = random.randint(0,x.shape[0])
      else:
        j = i
      plt.imshow(x[j].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()
  
plot_mnist(x_test_noisy, y_test, randomly=True)

# flatten the 28x28 images into vectors of size 784.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_train_noisy = x_train_noisy.reshape((len(x_train_noisy), np.prod(x_train_noisy.shape[1:])))
x_test_noisy = x_test_noisy.reshape((len(x_test_noisy), np.prod(x_test_noisy.shape[1:])))

#training
history = autoencoder.fit(x_train_noisy, x_train,
                          epochs=100,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test_noisy, x_test))
                          
# plot training performance
def plot_training_loss(history):

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(1, len(loss) + 1)

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()

plot_training_loss(history)

# plot de-noised images
def plot_mnist_predict(x_test, x_test_noisy, autoencoder, y_test, labels=[]):
  
  if len(labels)>0:
    x_test = x_test[np.isin(y_test, labels)]
    x_test_noisy = x_test_noisy[np.isin(y_test, labels)]

  decoded_imgs = autoencoder.predict(x_test)
  n = 10  
  plt.figure(figsize=(20, 4))
  for i in range(n):
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(x_test_noisy[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(decoded_imgs[i].reshape(28, 28))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()
  return decoded_imgs, x_test
 
decoded_imgs_test, x_test_new = plot_mnist_predict(x_test, x_test_noisy, autoencoder, y_test)
