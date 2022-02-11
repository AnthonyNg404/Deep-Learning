import random

import numpy as np

import torch
import torch.nn.functional as F

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Construct new tensor that requires gradient computation
    X = X.clone().detach().requires_grad_(True)
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with torch.autograd.gard.           #
    ##############################################################################
    scores = model(X)
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    scores.backward(torch.ones(scores.size()))

    dX = X.grad
    dX = dX.abs()
    dX, _ = torch.max(dX, dim=1)

    saliency = dX

    '''

    model = model.cuda()
    X = X.cuda().detach().requires_grad_(True)
    y = y.cuda()

    scores = model(X)
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    scores.backward(torch.ones(scores.size()).cuda())

    dX = X.grad
    dX = dX.abs()
    dX, _ = torch.max(dX, dim=1)

    saliency = dX.cpu()
    '''
    pass
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image.
    X_fooling = X.clone().detach().requires_grad_(True)

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    scores = model(X_fooling)
    while torch.argmax(scores) != target_y:
      scores = scores.gather(1, torch.tensor(target_y).view(-1, 1)).squeeze()
      scores.backward(torch.ones(scores.size()))   

      dX = X_fooling.grad
      dX = learning_rate * dX / torch.norm(dX)
      with torch.no_grad():
        X_fooling += dX
        X_fooling.grad.zero_()
      scores = model(X_fooling)
    '''

    model = model.cuda()
    X_fooling = X.cuda().detach().requires_grad_(True)
    target_y = torch.tensor(target_y).cuda()

    scores = model(X_fooling)
    while torch.argmax(scores) != target_y:
      scores = scores.gather(1, target_y.view(-1, 1)).squeeze()
      scores.backward(torch.ones(scores.size()).cuda())   

      dX = X_fooling.grad
      dX = learning_rate * dX / torch.norm(dX)
      with torch.no_grad():
        X_fooling += dX
        X_fooling.grad.zero_()
      scores = model(X_fooling)

    X_fooling = X_fooling.cpu()
    model = model.cpu()
    '''
    pass
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling.detach()


def update_class_visulization(model, target_y, l2_reg, learning_rate, img):
    """
    Perform one step of update on a image to maximize the score of target_y
    under a pretrained model.

    Inputs:
    - model: A pretrained CNN that will be used to generate the image
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - img: the image tensor (1, C, H, W) to start from
    """

    # Create a copy of image tensor with gradient support
    img = img.clone().detach().requires_grad_(True)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    scores = model(img)
    scores = scores.gather(1, torch.tensor(target_y).view(-1, 1)).squeeze()
    scores -= l2_reg * torch.norm(img)**2
    scores.backward(torch.ones(scores.size()))   

    dimg = img.grad
    with torch.no_grad():
      img += learning_rate * dimg
      img.grad.zero_()
    '''

    img = img.cpu()
    model = model.cpu()

    model = model.cuda()
    img = img.cuda().detach().requires_grad_(True)
    target_y = torch.tensor(target_y).cuda()

    scores = model(img)
    scores = scores.gather(1, target_y.view(-1, 1)).squeeze()
    scores -= l2_reg * torch.norm(img)**2
    scores.backward(torch.ones(scores.size()).cuda())   

    dimg = img.grad
    with torch.no_grad():
      img += learning_rate * dimg
      img.grad.zero_()

    img = img.cpu()
    model = model.cpu()
    '''
    pass
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img.detach()
