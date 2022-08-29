import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import codecs
import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True
import numpy as np
import gradio as gr
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Get model weights
os.system("wget https://huggingface.co/hasibzunair/melanet/resolve/main/MelaNet.h5")

# Load model
model = None
model = load_model("MelaNet.h5", compile=False)
model.summary()

# Class label list
labels = ["Benign", "Malignant"]

# Helpers
def preprocess_image(img_array):
    # Normalize to [0,1]
    img_array = img_array.astype("float32")
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
def inference(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = preprocess_image(img)
    img = np.expand_dims(img, 0)
    preds = model.predict(img)
    # Predict
    preds = model.predict(img)[0]
    labels_probs = {labels[i]: float(preds[i]) for i, _ in enumerate(labels)}
    return labels_probs


title = "Melanoma Detection using Adversarial Training and Deep Transfer Learning"
description = codecs.open("description.html", "r", "utf-8").read()
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2004.06824' target='_blank'>Melanoma Detection using Adversarial Training and Deep Transfer Learning</a> | <a href='https://github.com/hasibzunair/adversarial-lesions' target='_blank'>Github</a></p>"

demo = gr.Interface(
    fn=inference,
    title=title,
    description=description,
    article=article,
    inputs=gr.inputs.Image(type="filepath", label="Input"),
    outputs="label",
    examples=[f"examples/{fname}.png" for fname in ["benign", "malignant"]],
    allow_flagging="never",
    analytics_enabled=False,
)

demo.launch()
