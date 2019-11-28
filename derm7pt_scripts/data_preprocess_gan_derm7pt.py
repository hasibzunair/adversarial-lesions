import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys



'''Divide BCC NEV and MEL in three folders for GAN training'''

def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

        
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  
import helpers

def get_data(path):
    img=cv2.imread(path)
    return img


# Training file directory
DATASET = os.path.join(ROOT_DIR, 'dataset')
DATASET_NUMPY = os.path.join(ROOT_DIR, 'dataset', "derm7ptnumpy")
DATASET_DERM7PT = os.path.join(ROOT_DIR, 'dataset', 'derm7ptnumpy')


helpers.create_directory("{}/derm7ptgan/".format(DATASET))
NEW_DATASET_PATH = "{}/{}".format(DATASET, "derm7ptgan")
print(NEW_DATASET_PATH)

helpers.create_directory("{}/bcc".format(NEW_DATASET_PATH))
helpers.create_directory("{}/nev".format(NEW_DATASET_PATH))
helpers.create_directory("{}/mel".format(NEW_DATASET_PATH))

# new paths
bcc_path = "{}/bcc".format(NEW_DATASET_PATH)
nev_path = "{}/nev".format(NEW_DATASET_PATH)
mel_path = "{}/mel".format(NEW_DATASET_PATH)

print(bcc_path, nev_path, mel_path)

x_train = np.load("{}/x_train.npy".format(DATASET_NUMPY))
y_train = np.load("{}/y_train.npy".format(DATASET_NUMPY))

y_train = np.array([np.argmax(y) for y in y_train])
print(x_train.shape, y_train.shape)


# save in folders
number = 0

for img, label in tqdm(zip(x_train, y_train)):
    
    if label == 0:
        # bcc
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(bcc_path + '/bcc' + str(number) + ".jpeg", img)
        img=None
    elif label == 1:
        # nev
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(nev_path + '/nev' + str(number) + ".jpeg", img)
        img=None
    elif label == 2:
        # mel
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(mel_path + '/mel' + str(number) + ".jpeg", img)
        img=None
    else:
        pass
    
    number+=1

print("Done!")
