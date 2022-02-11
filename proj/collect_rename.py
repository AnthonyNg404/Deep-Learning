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
import shutil


output_dir = "C:/Users/antho/Desktop/collection/"
input_dir = "C:/Users/antho/Desktop/image/image/"
input_floders = os.listdir(input_dir)

def main():
    g = 0
    for floder in input_floders:
        print(g)
        images = glob.glob(input_dir + '/' + floder + '//*jpg')
        for image in images:
            shutil.copyfile(image, output_dir + '//' + floder + '#' + image.split('\\')[-1])
        g+=1

main()


