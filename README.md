# [Melanoma Detection using Adversarial Training and Deep Transfer Learning](https://iopscience.iop.org/article/10.1088/1361-6560/ab86d3)

## aka How to make skin lesions cancerous (in images) with Unpaired Image Translation

This code is part of the supplementary materials for our paper titled *Melanoma Detection using Adversarial Training and Deep Transfer Learning* accepted for publication in the Journal of Physics in Medicine and Biology (PMB).

Full text of accepted version avaiable at [RG](https://www.researchgate.net/publication/340467505_Melanoma_detection_using_adversarial_training_and_deep_transfer_learning). 

Authors: Hasib Zunair and A. Ben Hamza

**TL;DR** Interclass variation is considered an intimidating remark in medical image analysis. Here we demonstrate an opposite perspective: we learn the interclass mappings(benign to malignant) and boost the minority class(malignant) using the orginal benign images and then train the classification model.

<p align="left">
<a href="#"><img src="media/visuals.png" width="75%"></a>
</p>


### Citation

```
To cite this article before publication: Hasib Zunair et al 2020 Phys. Med. Biol. In press https://doi.org/10.1088/1361-6560/ab86d3
```


### Major Requirements

This code requires

* Python: 3.6
* Tensorflow: 2.0.0
* Keras: 2.3.1

A conda file will be added soon.


### Preparing training and test datasets

* Download dataset from https://challenge.kitware.com/#phase/5667455bcad3a56fac786791
* Clone this repo (obviously!)
* In this directory, make a folder in `dataset` named `isic2016` and keep all files there
* To build training set and test set

```python data_process_isic2016.py```

* To sort images for training CycleGAN (two folders malignant and benign)

```python data_process_gan.py``` 



### Training (both stages)

Now that the data is partitioned according to its class label (benign and malignant), we train CycleGAN on this data.

* Run `train_cyclegan.ipynb`

For generating the minority class (malignant) using the benign samples using the translation model:

* Run `upsampler.ipynb` to oversample and balance the dataset

Train the classification model using the oversampled and balanced dataset

* Train classifier using `train_ISIC_2016.ipynb` 

### Evaluation

The notebook `train_ISIC_2016.ipynb` consists the code to evalute on the ISIC 2016 test set.


### Pre-trained weights

Weight file for both the translation and classification model is made available under releases. 

| Models | Weights|
|:---:|:---:|
| Generator | [isic2016_benign_to_malignant_cyclegan_generator.h5](https://github.com/hasibzunair/adversarial-lesions/releases/latest/download/isic2016_benign_to_malignant_cyclegan_generator.h5) |
| MelaNet | [MelaNet.h5](https://github.com/hasibzunair/adversarial-lesions/releases/latest/download/MelaNet.h5) |


### Results

We report the AUC, Sensitivity and False negatives on the ISIC 2016 test set.

<p align="left">
<a href="#"><img src="media/test_results.png" width="85%"></a>
</p>

UMAP embeddings of the original benign and malignant lesions (left) and with the generated malignant lesions (right).

<p align="left">
<a href="#"><img src="media/umap.png" width="70%"></a>
</p>


### Reference to other models

Model definition scripts are based on https://github.com/eriklindernoren/Keras-GAN/tree/master/cyclegan and https://github.com/qubvel/classification_models.


### License

Your driver's license.
