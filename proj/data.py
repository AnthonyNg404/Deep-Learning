import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import random
import sys
import os
from os.path import basename, normpath
import glob
import time
import csv
import scipy
import numpy
import shutil

images = glob.glob("C:/Users/antho/Desktop/photos_all/*jpg")

for img_path in images:
    with Image.open(img_path) as img:
        h, w = img.size
    if h/w > 2 or h/w < 0.5:
        os.remove(img_path)

