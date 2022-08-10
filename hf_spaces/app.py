import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
import numpy as np
import gradio as gr
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Get model weights
os.system("wget https://github.com/hasibzunair/adversarial-lesions/releases/latest/download/MelaNet.h5")

# Load model
model = None
model = load_model("MelaNet.h5", compile=False)
model.summary()

# Path to examples and class label list
examples = ["benign.png", "malignant.png"]
labels = ["Benign", "Malignant"]

# Helpers
def preprocess_image(img_array):
    # Normalize to [0,1]
    img_array = img_array.astype('float32')
    img_array /= 255
    # Check that images are 2D arrays
    if len(img_array.shape) > 2:
        img_array = img_array[:, :, 0]
    # Convert to 3-channel
    img_array = np.stack((img_array, img_array, img_array), axis=-1)
    # Convert to array
    img_array = cv2.resize(img_array, (256, 256))
    return img_array


# Main inference function
def inference(img):
    img = preprocess_image(img)
    img = np.expand_dims(img, 0)
    preds = model.predict(img)
    # Predict
    preds = model.predict(img)[0]
    labels_probs = {labels[i]: float(preds[i]) for i, _ in enumerate(labels)}
    return labels_probs

title = "Melanoma Detection Demo"
description = "This model predicts if the given image has benign or malignant symptoms. To use it, simply upload a skin lesion image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2004.06824' target='_blank'>Melanoma Detection using Adversarial Training and Deep Transfer Learning</a> | <a href='https://github.com/hasibzunair/adversarial-lesions' target='_blank'>Github</a></p>"

gr.Interface(
    fn=inference,
    title=title,
    description = description,
    article=article,
    inputs="image",
    outputs="label",
    examples=examples,
).launch(debug=True, enable_queue=True)