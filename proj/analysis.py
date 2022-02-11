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


res18 = "C:/Users/antho/Desktop/photos/output_res18.csv"
res50 = "C:/Users/antho/Desktop/photos/output_res50.csv"
res152 = "C:/Users/antho/Desktop/photos/output_res152.csv"
vgg = "C:/Users/antho/Desktop/photos/output_vgg19_bn.csv"
dense = "C:/Users/antho/Desktop/photos/output_densenet161.csv"

one_agreed = "C:/Users/antho/Desktop/agreed/1-agreed"
two_agreed = "C:/Users/antho/Desktop/agreed/2-agreed"
three_agreed = "C:/Users/antho/Desktop/agreed/3-agreed"

photos = glob.glob("C:/Users/antho/Desktop/photos/*jpg")
photos_dir = "C:/Users/antho/Desktop/photos"
imagenet_dir = "C:/Users/antho/Desktop/agreed/imagenet1.txt"



def collect(argeed):

    shutil.copyfile(image, output_dir + '//' + floder + '#' + image.split('\\')[-1])


def main():
    imagenet_dict = {}
    with open(imagenet_dir) as f:
        for line in f:
            (key, val) = line.split(":")
            print(type(int(key)))
            imagenet_dict[int(key[1:])] = val[:-2]

    print(imagenet_dict.keys())

    with open(res18, mode='r') as infile:
        reader = csv.reader(infile)
        with open('coors_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            res18_dict = {rows[0]: rows[1] for rows in reader}

    with open(res50, mode='r') as infile:
        reader = csv.reader(infile)
        with open('coors_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            res50_dict = {rows[0]: rows[1] for rows in reader}

    with open(res152, mode='r') as infile:
        reader = csv.reader(infile)
        with open('coors_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            res152_dict = {rows[0]: rows[1] for rows in reader}

    with open(dense, mode='r') as infile:
        reader = csv.reader(infile)
        with open('coors_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            dense_dict = {rows[0]: rows[1] for rows in reader}

    with open(vgg, mode='r') as infile:
        reader = csv.reader(infile)
        with open('coors_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            vgg_dict = {rows[0]: rows[1] for rows in reader}

    print(len(res18_dict.keys()))
    print(len(res50_dict.keys()))
    print(len(res152_dict.keys()))
    print(len(dense_dict.keys()))
    print(len(vgg_dict.keys()))

    five_ag = 0
    four_ag = 0
    three_ag = 0
    two_ag = 0
    one_ag = 0
    argeed = {}

    for name in res18_dict.keys():
        if res152_dict[name] == dense_dict[name] == vgg_dict[name]:
            three_ag += 1
            argeed[name] = 3
            index = int(res152_dict[name])
            #print(type(index))
            cate = imagenet_dict[index]
            shutil.copyfile(photos_dir + '//' + name, three_agreed + '//' + str(index) + '#' + str(cate) + '#' + name)

        elif res152_dict[name] == dense_dict[name]:
            two_ag += 1
            argeed[name] = 2
            index = int(res152_dict[name])
            cate = imagenet_dict[index]
            shutil.copyfile(photos_dir + '//' + name, two_agreed + '//' + str(index) + '#' + str(cate) + '#' + name)

        elif res152_dict[name] == vgg_dict[name]:
            two_ag += 1
            argeed[name] = 2
            index = int(res152_dict[name])
            cate = imagenet_dict[index]
            shutil.copyfile(photos_dir + '//' + name, two_agreed + '//' + str(index) + '#' + str(cate) + '#' + name)

        elif dense_dict[name] == vgg_dict[name]:
            two_ag += 1
            argeed[name] = 2
            index = int(dense_dict[name])
            cate = imagenet_dict[index]
            shutil.copyfile(photos_dir + '//' + name, two_agreed + '//' + str(index) + '#' + str(cate) + '#' + name)

        else:
            one_ag += 1
            argeed[name] = 1
            shutil.copyfile(photos_dir + '//' + name, one_agreed + '//' + 'No_agreed#' + name)

    print(three_ag, "  3ag")
    print(two_ag, "   2ag")
    print(one_ag, "   1ag")

main()