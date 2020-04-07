# Melanoma Detection using Adversarial Training and Deep Transfer Learning

## aka How to make skin lesions cancerous (in images) with Unpaired Image Translation

This code is part of the supplementary materials for our paper titled *Melanoma Detection using Adversarial Training and Deep Transfer Learning* accepted for publication in the Journal of Physics in Medicine and Biology (PMB).

Authors: Hasib Zunair and A. Ben Hamza

**TL;DR** Interclass variation is considered an intimidating remark in medical image analysis. Here we demonstrate an opposite perspective: we leverage this aspect to train a translation model to learn the interclass mappings(benign to malignant) and boost the minority class(malignant) using the orginal benign images and then train the classification model.


<p align="center">
<a href="#"><img src="media/visuals.png" width="75%"></a>
</p>

Paper details will be added soon.

### Major Requirements

* Python: 3.6
* Tensorflow: 2.0.0
* Keras: 2.3.1


### Getting started

* Download dataset from https://challenge.kitware.com/#phase/5667455bcad3a56fac786791
* Clone this repo (obviously!)
* In this directory, make a folder in `dataset` named `isic2016` and keep all files there
* Run `data_process_isic2016.py` to make training set
* Run `data_process_gan.py` to get data for training CycleGAN (two folders malignant and benign)
* Run `train_cyclegan.ipynb` to train CycleGAN 
* Run `upsampler.ipynb` to oversample the minority class and balance the dataset
* Train classifier using `train_ISIC_2016.ipynb` and evalute on the ISIC 2016 test set

### Models
Weight file for both the translation and classification models will be made available soon. (Weight files are in my lab's secondary computer and I cannot excess remotely nor can I go due to pandemic!)

### License

Your driver's license.
