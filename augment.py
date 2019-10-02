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
x_train = np.load("{}/x_baln.npy".format(dataset_path))
y_train = np.load("{}/y_baln.npy".format(dataset_path))

seq = iaa.Sequential([
    iaa.ContrastNormalization((0.5, 1.5)),
    
    iaa.Crop(percent=(0, 0.2)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.Sometimes(0.7, 
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
    ),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    #iaa.Multiply((0.8, 1.2), per_channel=0.2),
    
    iaa.Affine(
        rotate=(-25, 25),
    ),
    iaa.Affine(
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    ),
    iaa.Affine(
        shear=(-25, 25)
    ),
    
    iaa.Sometimes(0.8, 
        iaa.CoarseDropout(0.03, size_percent=0.1)
    ),
    iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ]),
    
    
], random_order=True) # apply augmenters in random order

def augment_data_minimal( x_values, y_values ):
    counter = 0
    RESIZE_DIM = 299
    X_values_augmented = []
    Y_values_augmented = []
    for x in x_values:
        for p in range(5):
            
            # seq 1
            Y_values_augmented.append( y_values[counter] )
            images_aug = seq.augment_images(x.reshape(1,RESIZE_DIM,RESIZE_DIM,3))   
            X_values_augmented.append( images_aug.reshape(RESIZE_DIM,RESIZE_DIM,3))
        counter = counter + 1
    
    # Quick math!
    # prev number of images = n
    # augmented number of images = n * 4 ( 2 seq 2 times)
    
    X_values_augmented = np.asarray( X_values_augmented )
    Y_values_augmented = np.asarray( Y_values_augmented )
    return (X_values_augmented, Y_values_augmented)


(x_aug, y_aug) = augment_data_minimal( x_train, y_train)
print(x_aug.shape, y_aug.shape)

x_train_aug = np.concatenate( (x_train, x_aug), axis = 0)
y_train_aug = np.concatenate( (y_train, y_aug), axis = 0)

print(x_train_aug.shape, y_train_aug.shape)
np.save("dataset/ISIC2016/x_augmented.npy", x_train_aug)
np.save("dataset/ISIC2016/y_augmented.npy", y_train_aug)
print("Done!")