import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils


# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  
import helpers

'''Save data in numpy format'''

def crop_and_resize(img, resize_dim=256):
    img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA)
    return img

def get_data(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=crop_and_resize(img)
    return img


# Raw dataset path
DATASET = os.path.join(ROOT_DIR, 'dataset')


# Training images path
TRAINING_IMAGES = os.path.join(DATASET, 'isic2016', 'ISBI2016_ISIC_Part3_Training_Data')
# Ground truth path
TRAINING_GT = os.path.join(DATASET, 'isic2016', 'ISBI2016_ISIC_Part3_Training_GroundTruth.csv')
# Read the metadata
TRAINING_META = pd.read_csv(TRAINING_GT, sep=',', names=["FILENAME", "CLASS"])

# Test images path
TEST_IMAGES = os.path.join(DATASET, 'isic2016', 'ISBI2016_ISIC_Part3_Test_Data')
# Ground truth path
TEST_GT = os.path.join(DATASET, 'isic2016', 'ISBI2016_ISIC_Part3_Test_GroundTruth.csv')
# Read the metadata
TEST_META = pd.read_csv(TEST_GT, sep=',', names=["FILENAME", "CLASS"])


def construct_numpy(images, meta, fname, lname):
    '''
    Creates a new numpy arrays.
    INPUT
        IMAGES: 
        df:
    OUTPUT
        Numpy arrays
    '''
    # filenames and gts
    filenames = meta['FILENAME'].values
    gt = meta['CLASS'].values
    
    # convert string labels to numeric values
    labels = []
    for s in gt:
        if s == "benign" or s == 0.0 :
            labels.append(0)
        if s == "malignant" or s == 1.0:
            labels.append(1)
            
    # all training images and labels     
    inp_feat = []
    g_t = []

    # two classes individually
    cancer = []
    non_cancer = []

    for f, l in tqdm(zip(filenames[:], labels[:])):
        f = "{}/{}.jpg".format(images, f)
        img = get_data(f)
        inp_feat.append(img)
        g_t.append(l)
        
        #----------------
        #if l == 1:
        #    cancer.append(img)
        #if l == 0:
        #    non_cancer.append(img)
        #else:
        #    pass
        #-----------------
        
        img = None

    # make nummpy arrays
    inp_feat = np.array(inp_feat)
    g_t = np.array(g_t)
    
    # one hot encoded vectors
    num_classes = 2
    g_t = np_utils.to_categorical(g_t,num_classes)

    #cancer = np.array(cancer)
    #non_cancer = np.array(non_cancer)

    print(inp_feat.shape, g_t.shape)
    
    # Create directory
    helpers.create_directory("dataset/isic2016numpy/")
    # Save
    np.save("dataset/isic2016numpy/{}.npy".format(fname), inp_feat)
    np.save("dataset/isic2016numpy/{}.npy".format(lname), g_t)
    
    print("Done!")
    


if __name__ == '__main__':
    
    # Make numpy arrays
    print("Training data...")
    construct_numpy(TRAINING_IMAGES, TRAINING_META, "x_train", "y_train")
    print("Test data...")
    construct_numpy(TEST_IMAGES, TEST_META, "x_test", "y_test")
    
    
    
    