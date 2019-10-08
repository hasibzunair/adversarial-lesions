import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.model_selection import train_test_split
import keras
from classification_models.keras import Classifiers

#!pip install image-classifiers==0.2.2
#!pip install image-classifiers==1.0.0b1

# load data
x_train = np.load("x_baln.npy")
y_train = np.load("y_baln.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

flag = 1
if flag == 1:
    
    # Shuffle data
    print("Shuffling data")
    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]
else:
    print("Not shuffling...")
    pass

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

arch, preprocess_input = Classifiers.get('inceptionresnetv2')

# preprocess data
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

x_train.shape, x_test.shape

# build model
base_model = arch(input_shape=(299,299,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(2, activation='softmax', trainable=True)(x)

#for layer in base_model.layers[:]:
#  layer.trainable=False

model = None
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

model.summary()


from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train, axis=1)), np.argmax(y_train, axis=1))
class_weights

EXP_NAME = "incepres_le-2__bcloss"


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    #eaturewise_center=False,
    #eaturewise_std_normalization=False,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest'
    )

datagen.fit(x_train)


adadelta = optimizers.Adadelta(lr=1e-3, rho=0.95)

model.compile(optimizer=adadelta, loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, 
                    epochs=200,
                    class_weight = class_weights,
                    verbose=1, 
                    validation_data=(x_test,y_test)
                   )


#convert ground truths to column values
y_test_flat = np.argmax(y_test, axis=1)
# make predictions
y_pred = model.predict(x_test, verbose=1)
# get labels from predictions
y_pred_flat = np.array([np.argmax(pred) for pred in y_pred])

# accuracy
print(acc(y_test_flat, y_pred_flat))
# average precision
from sklearn.metrics import average_precision_score
average_precision_score(y_test_flat, y_pred_flat)
# area under curve
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test_flat, y_pred_flat)
print(metrics.auc(fpr, tpr))

# all metrics report
from sklearn.metrics import confusion_matrix, classification_report

confusion_mtx = confusion_matrix(y_test_flat, y_pred_flat) 
print(confusion_mtx)

target_names = ['0', '1']
print(classification_report(y_test_flat, y_pred_flat, target_names=target_names))


from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_flat, y_pred_flat)
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)