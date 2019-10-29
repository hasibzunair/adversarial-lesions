import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm
import helpers


def get_data(path):
    img=cv2.imread(path)
    return img


#Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Training file directory
DATASET = os.path.join(ROOT_DIR, 'dataset')

helpers.create_directory("{}/isic2016gan/".format(DATASET))
NEW_DATASET_PATH = "{}/{}".format(DATASET, "isic2016gan")
NEW_DATASET_PATH

helpers.create_directory("{}/normal".format(NEW_DATASET_PATH))
helpers.create_directory("{}/cancer".format(NEW_DATASET_PATH))

# new paths
CANCER_FOLDER = "{}/cancer".format(NEW_DATASET_PATH)
NORMAL_FOLDER = "{}/normal".format(NEW_DATASET_PATH)

CANCER_FOLDER, NORMAL_FOLDER


# IMAGES PATH
TRAINING_IMAGES = os.path.join(DATASET, 'isic2016', 'ISBI2016_ISIC_Part3_Training_Data')
# GROUND TRUTH PATH
TRAINING_GT = os.path.join(DATASET, 'isic2016', 'ISBI2016_ISIC_Part3_Training_GroundTruth.csv')
# Read the metadata
TRAINING_META = pd.read_csv(TRAINING_GT, sep=',', names=["FILENAME", "CLASS"])

# filenames and gts
filenames = TRAINING_META['FILENAME'].values
gt = TRAINING_META['CLASS'].values
    
# convert string labels to numeric values
labels = []
for s in gt:
    if s == "benign" or s == 0.0 :
        labels.append(0)
    if s == "malignant" or s == 1.0:
        labels.append(1)

# save in folders
number = 0  
for f, l in tqdm(zip(filenames[:], labels[:])):
    f = "{}/{}.jpg".format(TRAINING_IMAGES, f)
    img = get_data(f)
    
    if l == 0.0:
            cv2.imwrite(NORMAL_FOLDER + '/normal' + str(number) + ".jpeg", img)
            img=None
    else:
        cv2.imwrite(CANCER_FOLDER + '/cancer' + str(number) + ".jpeg", img)
        img=None
    number+=1

print("Done!")