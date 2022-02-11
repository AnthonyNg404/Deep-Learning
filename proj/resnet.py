import logging
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import random


num_epochs = 30
l_r = 0.001
batch_size = 64

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_1, stride_2):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Sequential(      # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=in_channels,              # input height
                out_channels=out_channels,            # n_filters
                kernel_size=kernel_size,              # filter size
                stride=stride_1,
                padding=(1, 1)                # filter movement/step                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)                     # activation
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(      # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=out_channels,              # input height
                out_channels=out_channels,            # n_filters
                kernel_size=kernel_size,              # filter size
                stride=stride_2,
                padding=(1, 1)                   # filter movement/step                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)                     # activation
            nn.BatchNorm2d(out_channels), 
        )   # fully connected layer, output 10 classes
        self.relu = nn.ReLU()
        self.stride_1 = stride_1
        if self.stride_1 == 2:
            self.input_reformat = nn.Conv2d(
                in_channels=in_channels,              # input height
                out_channels=out_channels,            # n_filters
                kernel_size=1,              # filter size
                stride=2,             # filter movement/step                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            )

    def forward(self, x):
        output = self.conv1(x)
        output_conv = self.conv2(output)
        x_map = x
        if self.stride_1 == 2:
            x_map = self.input_reformat(x)
        output = self.relu(x_map + output_conv)
        return (output, output_conv)

class Resnet(nn.Module):
    def __init__(self, num_classes):
        super(Resnet, self).__init__()
        self.conv1 = nn.Sequential(      # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=64,            # n_filters
                kernel_size=3,              # filter size = 7 for original resnet-18
                stride=2,                   # filter movement/step                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)                     # activation
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(3, stride=2)
        self.block_64_1 = ResnetBlock(64, 64, 3, 1, 1)
        #self.block_64_2 = ResnetBlock(64, 64, 3, 1, 1)
        self.block_128_1 = ResnetBlock(64, 128, 3, 2, 1)
        #self.block_128_2 = ResnetBlock(128, 128, 3, 1, 1)
        self.block_256_1 = ResnetBlock(128, 256, 3, 2, 1)
        #self.block_256_2 = ResnetBlock(256, 256, 3, 1, 1)
        self.block_512_1 = ResnetBlock(256, 512, 3, 2, 1)
        #self.block_512_2 = ResnetBlock(512, 512, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(512, 1000)
        self.out_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.max_pool(output)
        output = self.block_64_1(output)[0]
        #output = self.block_64_2(output)
        output = self.block_128_1(output)[0]
        #output = self.block_128_2(output)
        output_2 = self.block_256_1(output)
        #output = self.block_256_2(output)
        output = self.block_512_1(output_2[0])[0]
       # output = self.block_512_2(output)
        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
        output_tsne = self.fully_connected(output) 
        output = self.out_layer(output_tsne) 
        return output
        #return output, output_2[1], output_tsne
        #return torch.nn.functional.softmax(output)


def train_and_validate_prob4():
    logging.basicConfig(filename='loss_prob4_1.txt', level=logging.DEBUG)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    val_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)

    resnet = Resnet(num_classes)

    optimizer = torch.optim.Adam(resnet.parameters(), lr=l_r) 

    loss_func = nn.CrossEntropyLoss()  

    for i in range(num_epochs):
        val_data_iter = iter(val_loader)
        for step, (train_features, train_labels) in enumerate(train_loader):
            train_out = resnet(train_features)
            #train_labels = torch.nn.functional.one_hot(train_labels, num_classes)
            loss = loss_func(train_out, train_labels)     
            optimizer.zero_grad()      
            loss.backward() 
            optimizer.step() 
            print(str(step) + ": " + str(loss))
            logging.info(str(step) + ": " + str(loss))
            if step%50 == 0:
                resnet.eval()
                #torch.save(resnet.state_dict(), 'prob4_model_' + str(i) + '.pt')
                val_features, val_labels = val_data_iter.next()
                #val_labels = torch.nn.functional.one_hot(val_labels, num_classes)
                val_out = resnet(val_features)
                val_loss = loss_func(val_out, val_labels)
                print("val_loss " + str(step) + ": " + str(val_loss))
                logging.info("val_loss " + str(step) + ": " + str(val_loss))
                resnet.train()
        torch.save(resnet.state_dict(), 'prob4_model_' + str(i) + '.pt')

def get_rotation_data():
    #logging.basicConfig(filename='loss_prob5.txt', level=logging.DEBUG)
    transform_0 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data_0 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_0)
    train_length = len(train_data_0)
    train_data_0.targets = torch.tensor([0]*train_length)
    val_data_0 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_0)
    val_length = len(val_data_0)
    val_data_0.targets = torch.tensor([0]*val_length)

    transform_90 = transforms.Compose(
    [torchvision.transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data_90 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_90)
    train_length = len(train_data_90)
    train_data_90.targets = torch.tensor([1]*train_length)
    val_data_90 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_90)
    val_length = len(val_data_90)
    val_data_90.targets = torch.tensor([1]*val_length)

    transform_180 = transforms.Compose(
    [torchvision.transforms.Lambda(lambda x: transforms.functional.rotate(x, 180)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data_180 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_180)
    train_length = len(train_data_180)
    train_data_180.targets = torch.tensor([2]*train_length)
    val_data_180 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_180)
    val_length = len(val_data_180)
    val_data_180.targets = torch.tensor([2]*val_length)

    transform_270 = transforms.Compose(
    [torchvision.transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data_270 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_270)
    train_length = len(train_data_270)
    train_data_270.targets = torch.tensor([3]*train_length)
    val_data_270 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_270)
    val_length = len(val_data_270)
    val_data_270.targets = torch.tensor([3]*val_length)

    return torch.utils.data.ConcatDataset([train_data_0, train_data_90, train_data_180, train_data_270]), torch.utils.data.ConcatDataset([val_data_0, val_data_90, val_data_180, val_data_270])

def train_and_validate_prob5():
    logging.basicConfig(filename='loss_prob5.txt', level=logging.DEBUG)
    train_data, val_data = get_rotation_data()

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

    num_classes = 4

    resnet = Resnet(num_classes)

    optimizer = torch.optim.Adam(resnet.parameters(), lr=l_r) 

    loss_func = nn.CrossEntropyLoss()  

    for i in range(num_epochs):
        val_data_iter = iter(val_loader)
        for step, (train_features, train_labels) in enumerate(train_loader):
            train_out = resnet(train_features)
            #train_labels = torch.nn.functional.one_hot(train_labels, num_classes)
            loss = loss_func(train_out, train_labels)     
            optimizer.zero_grad()      
            loss.backward() 
            optimizer.step() 
            print(str(step) + ": " + str(loss))
            logging.info(str(step) + ": " + str(loss))
            if step%50 == 0:
                resnet.eval()
                val_features, val_labels = val_data_iter.next()
                #val_labels = torch.nn.functional.one_hot(val_labels, num_classes)
                val_out = resnet(val_features)
                val_loss = loss_func(val_out, val_labels)
                print("val_loss " + str(step) + ": " + str(val_loss))
                logging.info("val_loss " + str(step) + ": " + str(val_loss))
                resnet.train()
        torch.save(resnet.state_dict(), 'prob5_model_' + str(i) + '.pt')


def get_best_worst_images():
    #train_data, val_data = get_rotation_data()
    model = Resnet(4)
    model.load_state_dict(torch.load("prob5_model_16.pt"))
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    count = 0
    for f,y in train_data:
        for label in range(4):
            tag = str(count) + "_" + str(label)
            img = transforms.functional.rotate(f, 90*label)
            img.save("prob_5_images/" + tag + ".png")
            x = transform(img)
            x = x.unsqueeze(0)
            model_out = model(x)
            #y = torch.nn.functional.one_hot(torch.tensor(label), 4).unsqueeze(0)
            loss = loss_func(model_out, torch.tensor(label).unsqueeze(0))
            print(tag + ": " + str(loss))
        count+=1


def get_highest_and_lowest_loss_img():
    loss_data = open("img_rotation_loss.txt")
    loss = []
    for line in loss_data:
        img_tag_index = line.find(":")
        if img_tag_index == -1:
            continue
        img_tag = line[:img_tag_index]
        start = line.find("tensor(") + 7
        end = line.find(", grad_fn")
        loss_val = float(line[start:end])
        loss.append((loss_val, img_tag))
    loss.sort()
    print(loss)

def show_filters():
    model = Resnet(10)
    model.load_state_dict(torch.load("prob4_model_12.pt"))
    model.eval()
    kernels = model.conv1[0].weight.detach()
    print(kernels.size())
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    filter_img = torchvision.utils.make_grid(kernels, nrow = 8)
    plt.imshow(filter_img.permute(1, 2, 0))
    img = torchvision.utils.save_image(kernels, 'resnet_first_conv_layer.png' ,nrow = 8)

def get_cam(class_index):
    image = Image.open('prob_5_images/10_0.png')
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    x = transform(image).unsqueeze(0)
    model = Resnet(4)
    model.load_state_dict(torch.load("prob5_model_16.pt"))
    model.eval()
    out, feature_conv, _ = model(x)
    batch_size, nc, h, w = feature_conv.shape
    weight_fc = model.out_layer.weight.detach()
    #cam = weight_fc[class_index].dot(feature_conv.reshape((nc, h*w)))
    reshaped_feature_conv = feature_conv.reshape((h*w, nc))
    out = None
    for weight in weight_fc[class_index]:
        if out is None:
            out = weight*reshaped_feature_conv
        else:
            out += weight*reshaped_feature_conv

    #cam = torch.matmul(weight_fc[class_index], feature_conv.reshape((h*w, nc)))
    #cam = cam.reshape(512, 1000).detach()
    cam = out.detach()
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    plt.imshow(cam_img)
    fig = plt.gcf()
    fig.set_size_inches(8, 1)
    plt.axis('off')
    plt.savefig("resnet_10_0.png")


def plot_loss():
    step = 0
    loss_data = open("loss_prob4_1.txt")
    train_steps = []
    train_loss = []
    val_steps = []
    val_loss = []
    min_val_loss = 10
    min_val_step = 0
    for line in loss_data:
        start = line.find("tensor(") + 7
        end = line.find(", grad_fn")
        loss_val = float(line[start:end])
        if "val_loss" in line:
            if loss_val < min_val_loss:
                min_val_loss = loss_val
                min_val_step = step
            val_steps.append(step)
            val_loss.append(loss_val)
        else:
            train_steps.append(step)
            train_loss.append(loss_val)
        step+=1
        # if step > 50000:
        #     break
    plt.plot(train_steps, train_loss, label="train")
    plt.plot(val_steps, val_loss, label="validation")
    plt.ylim(0,2.0)
    plt.title("Cross Entropy Train/Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Cross Entropy")
    plt.legend()
    plt.show()
    plt.savefig("test_plot.png")
    print("min val:", min_val_loss)
    print("min val step:", min_val_step)

rotation_color = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'black'
}

image_category_color = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'black',
    4: 'orange',
    5: 'purple',
    6: 'brown',
    7: 'pink',
    8: 'gray',
    9: 'turquoise'
}

def model_tsne():
    all_layers = []
    model = Resnet(4)
    model.load_state_dict(torch.load("prob5_model_16.pt"))
    model.eval()
    colors = []
    photo_numbers = random.sample(range(4000), 25)
    for i in range(4):
        for photo_number in photo_numbers:
            image_file = 'prob_5_images/' + str(photo_number) + "_" + str(i) + ".png"
            image = Image.open(image_file)
            transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            x = transform(image).unsqueeze(0)
            last_layer_out = model(x)[2]
            last_layer_out = last_layer_out.squeeze(0).detach().numpy()
            all_layers.append(last_layer_out)
            colors.append(rotation_color[i])
    tsne_out = TSNE(n_components=2).fit_transform(all_layers)
    plt.scatter(tsne_out[:,0], tsne_out[:,1], c=colors)
    plt.title("TSNE for self supervised model")
    plt.savefig("tsne_self_supervised.png")

def model_tsne_for_resnet():
    all_layers = []
    model = Resnet(10)
    model.load_state_dict(torch.load("prob4_model_16.pt"))
    model.eval()
    colors = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    count = 0
    for x,y in train_data:
        if count > 100:
            break
        last_layer_out = model(x.unsqueeze(0))[2]
        last_layer_out = last_layer_out.squeeze(0).detach().numpy()
        all_layers.append(last_layer_out)
        colors.append(image_category_color[y])
        count +=1
    tsne_out = TSNE(n_components=2).fit_transform(all_layers)
    plt.scatter(tsne_out[:,0], tsne_out[:,1], c=colors)
    plt.title("TSNE for Resnet")
    plt.savefig("tsne_resnet.png")



def main():
    #plot_loss()
    train_and_validate_prob4()
    #get_best_worst_images()
    #get_highest_and_lowest_loss_img()
    #show_filters()
    #train_and_validate_prob4()
    #get_cam(2)
    #model_tsne()
    #model_tsne_for_resnet()

main()
