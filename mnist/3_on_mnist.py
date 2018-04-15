from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from mnist_model import MNIST_Net

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import utils

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=9, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--ra', type=int, default=3, metavar='N',
                    help='reassign interval')
parser.add_argument('--mlp_epochs', type=int, default=12, metavar='N',
                    help='no of epochs to train the fc layers')
parser.add_argument('--output-dim', type=int, default='50', metavar='N',
                    help='fc1 dimension')
parser.add_argument('--mlp-lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


output_dim = args.output_dim
train_set_size = 60000
test_set_size  = 10000
train_batch_size = 100


def train_sup(epoch,optimizer,train_images,train_labels):
    model.train()
    train_loss = 0
    correct = 0


    train_images, train_labels = utils.shuffle_together(train_images,train_labels)

    for i in range(0,train_set_size,train_batch_size):

        data = train_images[i:i+train_batch_size]
        target = train_labels[i:i+train_batch_size]

        batch_idx = i / float(train_batch_size)

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

    #for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_set_size,
                10000. * batch_idx / float(train_set_size), loss.data[0]))

    train_loss /= float(train_set_size)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, train_set_size,
        100. * correct / float(train_set_size)))




def train_unsup(epoch,optimizer,targets,train_images):
    model.train()
    for i in range(0,train_set_size,train_batch_size):

        #get this batch's data
        data = train_images[i : i + train_batch_size]
        data = torch.from_numpy(data).float()

        #get this batch's targets
        currTargets = targets[i : i + train_batch_size]
        currTargets_tens = torch.from_numpy(currTargets).float()


        #convert to variables
        data_var = Variable(data)
        currTargets_var = Variable(currTargets_tens)

        if args.cuda:
            data_var = data_var.cuda()
            currTargets_var = currTargets_var.cuda()


        optimizer.zero_grad()   #optimizer clear
        output = model.get_features(data_var)    #forward pass

        #if it's time to permute the assignments
        if (epoch-1) % args.ra == 0 :
            #permute the assignments
            batchTargets = utils.calc_optimal_target_permutation(output.data.cpu().numpy(),currTargets)
            #add those targets to the target list
            targets[i : i + train_batch_size] = batchTargets

        #get this batch's targets
        currTargets = targets[i : i + train_batch_size]
        currTargets_tens = torch.from_numpy(currTargets).float()

        loss = F.mse_loss(output,currTargets_var)   #calculate loss
        loss.backward()     #backward pass
        optimizer.step()     #update gradients

        if (i/train_batch_size) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i, train_set_size,
                100. * i / float(train_set_size), loss.data[0]))

    return targets


def test(test_images,test_labels):
    model.eval()
    test_loss = 0
    correct = 0

    test_images, test_labels = utils.shuffle_together(test_images, test_labels)

    for i in range(0,test_set_size,args.test_batch_size):

        data = test_images[i:i+args.test_batch_size]
        target = test_labels[i:i+args.test_batch_size]

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max logH[Z]

        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= test_set_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_set_size,
        100. * correct / test_set_size))




def freeze_conv_layers():
   for param in model.conv1.parameters():
     param.requires_grad = False

   for param in model.conv2.parameters():
     param.requires_grad = False



print('Initializing model..')
model = MNIST_Net(output_dim=output_dim)
if args.cuda:
    model.cuda()

#Generating targets
targets = utils.generateTargetReps(train_set_size,output_dim)


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

print(train_images.shape,test_images.shape)
print(train_labels.shape,test_labels.shape)





optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)

train_imgs = train_images
print('Training (unsupervised)..')
for epoch in range(1,args.epochs+1):
    if (epoch-1) % args.ra == 0:
        print('Shuffling the images,targets..')
        train_imgs, targets = utils.shuffle_together(train_imgs,targets)
    targets = train_unsup(epoch,optimizer,targets,train_imgs)

print('Got the features..')
print('Training with labels..')

print('Freezing convolutional layers')
#freezing the convolutional layers
freeze_conv_layers()




optimizer = optim.SGD([
		{'params': model.fc1.parameters()},
		{'params': model.fc2.parameters()}
		],lr=args.mlp_lr,momentum=args.momentum)

for epoch in range(1,args.mlp_epochs+1):
    train_sup(epoch,optimizer, train_images, train_labels)
    test(test_images,test_labels)
