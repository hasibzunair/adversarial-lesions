import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
from keras.utils.data_utils import get_file

def get_model():
    URL = "https://github.com/hasibzunair/adversarial-lesions/releases/latest/download/MelaNet.h5"
    weights_path = get_file(
                "MelaNet.h5",
                URL)
    model = load_model(weights_path, compile = False)
    return model