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

'''
crop to 1.5 WH/HW ratio then stretch
reads in file path 
returns tensor with size(1,3,224,224)
'''
def resize_crop_point_five(img_path):
    img = Image.open(img_path)
    if img.mode == 'L' or img.mode == 'CMYK' or img.mode == 'RGBA':
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        img = rgbimg
    w, h = img.size
    stretch = transforms.Resize((224, 224), interpolation=Image.BILINEAR)

    if h / w > 1.5 or w / h > 1.5:

        if h < w:
            crop_point_five = transforms.CenterCrop((h, round(h*1.5)))
        else:
            crop_point_five = transforms.CenterCrop((round(w*1.5), w))

        transform = transforms.Compose(
            [crop_point_five, stretch, transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return transform(img).view(1, 3, 224, 224)

    else:
        transform = transforms.Compose(
            [stretch, transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return transform(img).view(1, 3, 224, 224)


'''
crop to 1.2 WH/HW ratio then stretch
reads in file path 
returns tensor with size(1,3,224,224)
'''
def resize_crop_point_two(img_path):
    img = Image.open(img_path)
    if img.mode == 'L' or img.mode == 'CMYK' or img.mode == 'RGBA':
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        img = rgbimg
    w, h = img.size
    stretch = transforms.Resize((224, 224), interpolation=Image.BILINEAR)

    if h / w > 1.2 or w / h > 1.2:

        if h < w:
            crop_point_two = transforms.CenterCrop((h, round(h * 1.2)))
        else:
            crop_point_two = transforms.CenterCrop((round(w * 1.2), w))

        transform = transforms.Compose(
            [crop_point_two, stretch, transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return transform(img).view(1, 3, 224, 224)

    else:
        transform = transforms.Compose(
            [stretch, transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        return transform(img).view(1, 3, 224, 224)


'''
directly stretch
reads in file path 
returns tensor with size(1,3,224,224)
'''
def resize_direct(img_path):
    img = Image.open(img_path)
    if img.mode == 'L' or img.mode == 'CMYK' or img.mode == 'RGBA':
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        img = rgbimg
    stretch = transforms.Resize((224, 224), interpolation=Image.BILINEAR)

    transform = transforms.Compose(
        [stretch, transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    return transform(img).view(1, 3, 224, 224)


images = glob.glob("D:/2010_val/*jpeg")
output_file = "D:/2010_val/output_dense161.csv"
model1 = models.densenet161(pretrained=True)


def main():
    batch_size = 64
    count = 0
    dict = {}
    img_cache = None
    name_cache = []
    model1.eval()
    with torch.no_grad():
        for image in images:
            print(image)
            if count < batch_size:
                #print(image)
                #print(image.split('\\')[-1])

                name_cache.append(image.split('\\')[-1])

                #img = Image.open(image)
                #trans = transforms.ToPILImage()
                #trans1 = transforms.ToTensor()
                #img1 = trans(trans1(img))
                #resize2 = transforms.Resize((224, 224), interpolation=Image.NEAREST)
                #resize3 = transforms.Resize((224, 224), interpolation=Image.BICUBIC)
                #img_trans = resize1(img)

                tensor = resize_direct(image)

                if count == 0:
                    img_cache = tensor
                else:
                    img_cache = torch.cat((img_cache, tensor), 0)
                count += 1

            if count == batch_size or len(images) - len(dict) == count:
                #print(img_cache.size())
                #print(len(images))
                score = model1(img_cache)
                _, label = score.max(1)
                #print(label.size())
                #print(label.tolist())
                #print(name_cache)
                prediction = label.tolist()
                score = score.tolist()
                print(len(score), label.size(), len(name_cache))
                for i, j, k in zip(name_cache, prediction, score):
                    dict[i] = [j, k]
                print(len(dict))
                count = 0
                img_cache = None
                name_cache = []
                #break

        with open(output_file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dict.items():
                writer.writerow([key, value[0], value[1]])

main()


'''
def main():
    photo_file = os.listdir(root_dir_photo)
    for i in photo_file:
        if i[:-3] != 'jpg':
            photo_file.remove(i)
    print(photo_file[0])
    im = Image.open(root_dir_photo + "/" + photo_file[0])
    #im.show()
    trans1 = transforms.ToTensor()
    im1 = trans1(im)
    trans2 = transforms.ToPILImage()
    im2 = trans2(im1)
    st = time.clock()

    images = glob.glob("C:/Users/antho/Desktop/photos/*jpg")
    st1 = time.clock()
    print(st1 - st, '   read in')
    for image in images:
        img = Image.open(image)
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        img1 = trans(trans1(img))
        plt.imshow(trans(trans1(img)))
        st2 = time.clock()
        print(st2 - st1, '   transform')
        resize1 = transforms.Resize((224, 224), interpolation=Image.BILINEAR)
        resize2 = transforms.Resize((224, 224), interpolation=Image.NEAREST)
        resize3 = transforms.Resize((224, 224), interpolation=Image.BICUBIC)
        img1 = resize1(img1)
        #img1.show()
        st3 = time.clock()
        print(st3 - st2, '   resize')
        tensor1 = trans1(img1).view(1, 3, 224, 224)
        model1 = models.resnet18(pretrained=True)
        model2 = models.resnet50(pretrained=True)
        model3 = models.resnet152(pretrained=True)
        model1.eval()
        score = model1(tensor1)
        _, label = score.max(1)
        st4 = time.clock()
        print(st4 - st3, '   predict')
        print(label1.size())
        print(label1)
        print(torch.argmax(label1, dim=1))
        break

main()
'''
