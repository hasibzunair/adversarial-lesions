import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils
from imgaug import augmenters as iaa
import helpers

'''Augments whole dataset using imgaug sequence'''

#Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Training file directory
dataset_path = os.path.join(ROOT_DIR, 'dataset', 'ISIC2016')
model_path = os.path.join(ROOT_DIR, "models")


# load data
x_train = np.load("{}/x_train.npy".format(dataset_path))
y_train = np.load("{}/y_train.npy".format(dataset_path))

# Balance dataset

seq_standard = iaa.Sequential([
    iaa.Fliplr(0.5), 
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-360, 360))
], random_order=True)

def augment_data_minimal( x_values, y_values ):
    counter = 0
    RESIZE_DIM = 299
    X_values_augmented = []
    Y_values_augmented = []
    for x in x_values:
        for p in range(5):
            
            # seq 1
            Y_values_augmented.append( y_values[counter] )
            images_aug = seq_standard.augment_images(x.reshape(1,RESIZE_DIM,RESIZE_DIM,3))   
            X_values_augmented.append( images_aug.reshape(RESIZE_DIM,RESIZE_DIM,3))
        counter = counter + 1
    
    # Quick math!
    # prev number of images = n
    # augmented number of images = n * 4 ( 2 seq 2 times)
    
    X_values_augmented = np.asarray( X_values_augmented )
    Y_values_augmented = np.asarray( Y_values_augmented )
    return (X_values_augmented, Y_values_augmented)



(x_aug, y_aug) = augment_data_minimal( x_train, y_train)
x_aug.shape, y_aug.shape

x_train_aug = np.concatenate( (x_train, x_aug), axis = 0)
y_train_aug = np.concatenate( (y_train, y_aug), axis = 0)

np.save("dataset/ISIC2016/x_aug_5400.npy", x_train_aug)
np.save("dataset/ISIC2016/y_aug_5400.npy", y_train_aug)
print("Done!")