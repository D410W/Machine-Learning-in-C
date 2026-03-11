import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow_datasets as tfds
import numpy as np

def ds_to_numpy(ds):
  images = []
  labels = []
  
  for image, label in ds:
    images.append(image.numpy())
    labels.append(label.numpy())
  
  return np.array(images), np.array(labels)

train_ds, test_ds = tfds.load("mnist", split=['train', 'test'], as_supervised=True)

train_images, train_labels = ds_to_numpy(train_ds)
test_images, test_labels = ds_to_numpy(test_ds)

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

train_images.astype('float32').tofile('train_images')
test_images.astype('float32').tofile('test_images')

train_labels.astype('float32').tofile('train_labels')
test_labels.astype('float32').tofile('test_labels')

