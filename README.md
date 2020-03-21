## Melanoma Detection using Adversarial Training and Deep Transfer Learning

This code is part of the supplementary materials for our paper titled *Melanoma Detection using Adversarial Training and Deep Transfer Learning* accepted in the Journal of Physics in Medicine and Biology (PMB).

**TL;DR** Interclass variation is considered an intimidating remark in medical image analysis. Here we demonstrate an opposite perspective: rather than devising solutions against interclass variation, we shift our efforts to leverage interclass variation to improve melanoma detection.


### Get started

* Download dataset from https://challenge.kitware.com/#phase/5667455bcad3a56fac786791
* Clone this repo (obviously!)
* In this directory, make a folder in `dataset` named `isic2016` and keep all files there
* Run `data_process_isic2016.py` to make training set
* Run `data_process_gan.py` to get data for training CycleGAN (two folders malignant and benign)
* Run `train_cyclegan.ipynb` to train CycleGAN 
* Run `upsampler.ipynb` to oversample the minority class and balance the dataset
* Train classifier using `train_ISIC_2016.ipynb` and evalute on the ISIC 2016 test set


More details will be added soon.