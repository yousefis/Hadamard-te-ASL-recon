import inspect
import os

import numpy as np
import tensorflow as tf

from functions.loss.perceptual_loss.vgg.pretrained_vgg.vgg16 import Vgg16
def loadWeightsData(vgg16_npy_path=None):
    if vgg16_npy_path is None:
        path = inspect.getfile(Vgg16)
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.join(path, "vgg16.npy")
        vgg16_npy_path = path
        print(vgg16_npy_path)
    return np.load(vgg16_npy_path, encoding='latin1').item()