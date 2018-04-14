#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:25:16 2018

@author: ayushivishwakarma
"""

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
import os
import argparse

from models.lenet import LeNet
from utils import progress_bar
from torch.autograd import Variable

output_dim = 10
train_set_size = 60000
test_set_size  = 10000

train_batch_size = 100

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    #net = MobileNetV2()
    net = LeNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
def train_unsup(epoch,optimizer,train_loader,indices,targets):
    net.train()
    for batch_idx, (data, train_y) in enumerate(train_loader):
            #convert the data tensor to a variable
            data = Variable(data)
            if args.cuda:
                 data = data.cuda()
            #clear the gradients in the optimizer
            optimizer.zero_grad()
            #forward pass this batch of data
            output = net.get_features(data)

            #convert the output into numpy array
            output_arr = output.data.cpu().numpy()

            #re do the assignment every 'reassign_interval' epochs
            if (epoch-1) % args.ra == 0:

                if batch_idx == 0:
                     print('Clearing the previous assignments..')
                     #re initialize the indices array
                     indices = []
                #normalize the output features
                denominator = np.linalg.norm(output_arr,axis=1).reshape(-1,1)
                if denominator.all() == 0:
                    output_arr /= 1e-5
                else:
                    output_arr /= denominator

                #extract the targets from the targets array
                current_offset = batch_idx*train_batch_size
                target = targets[current_offset : current_offset + train_batch_size]

                #generate cost matrix
                cost_matrix = cdist(output_arr,target,metric='euclidean')

                #apply the hungarian algorithm to get the optimal assignment indices
                optimal_indices = linear_sum_assignment(cost_matrix)[1]

                #add the batch index * batch_size as offset
                optimal_indices += current_offset

                optimal_indices = optimal_indices.tolist()



                #add the optimal indices to the global indices array for further reference
                #indices = np.append(indices,optimal_indices)
                indices.extend(optimal_indices)


            #get the currently assigned targets to each feature vector based on the optimal assignment
            current_offset = batch_idx*train_batch_size
            assigned_indices = indices[current_offset : current_offset + train_batch_size]
            #assigned_indices = np.array(assigned_indices,dtype='int32')

            assigned_targets = targets[assigned_indices]

            #convert into Variable
            assigned_targets = assigned_targets.astype(float)
            assigned_targets_tens = torch.FloatTensor(assigned_targets)


            assigned_targets_var = Variable(assigned_targets_tens)

            if args.cuda:
                assigned_targets_var = assigned_targets_var.cuda()

            #found the error: gotta debug it, indices are not being stored for the next epoch, when batch_idx is 0 again

            loss = F.mse_loss(output,assigned_targets_var)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
               print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.data[0]))

    #Return the model and assignment list so it can be used later.
    return indices


def test():
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))




def freeze_conv_layers():
   for param in net.conv1.parameters():
     param.requires_grad = False

   for param in net.conv2.parameters():
     param.requires_grad = False


print('Initializing model..')




optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=args.momentum)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=train_batch_size, shuffle=False, **kwargs)

print('Training (unsupervised)..')

#Generating targets
targets = np.random.randn(train_set_size,output_dim)
#Normalize
targets /= np.linalg.norm(targets,axis=1).reshape(-1,1)

#indexes to store the assignments
indices = []
for epoch in range(1,args.epochs+1):
     indices = train_unsup(epoch,optimizer,train_loader,indices,targets)

print('Got the features..')
print('Training with labels..')

print('Freezing convolutional layers')
#freezing the convolutional layers
#model = freeze_conv_layers(model)


optimizer = optim.SGD([
		{'params': net.fc1.parameters()},
		{'params': net.fc2.parameters()}
		],lr=args.lr,momentum=args.momentum)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=train_batch_size, shuffle=True, **kwargs)



for epoch in range(1,args.mlp_epochs+1):
    train_unsup(epoch,optimizer,train_loader)
    test()

