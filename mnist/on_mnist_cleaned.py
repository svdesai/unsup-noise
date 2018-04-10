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



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=9, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


output_dim = 50
train_set_size = 60000
test_set_size  = 10000

train_batch_size = 100





kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)



model = MNIST_Net(output_dim=50)
if args.cuda:
    model.cuda()





def train_sup(epoch,optimizer,train_loader):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))







def train_unsup(epoch,optimizer,train_loader,indices,targets):
    model.train()
    for batch_idx, (data, train_y) in enumerate(train_loader):
            #convert the data tensor to a variable
            data = Variable(data)
            if args.cuda:
                 data = data.cuda()
            #clear the gradients in the optimizer
            optimizer.zero_grad()
            #forward pass this batch of data
            output = model.get_features(data)

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
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




def freeze_conv_layers():
   for param in model.conv1.parameters():
     param.requires_grad = False

   for param in model.conv2.parameters():
     param.requires_grad = False


print('Initializing model..')




optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)

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
		{'params': model.fc1.parameters()},
		{'params': model.fc2.parameters()}
		],lr=args.lr,momentum=args.momentum)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=train_batch_size, shuffle=True, **kwargs)



for epoch in range(1,args.mlp_epochs+1):
    train_sup(epoch,optimizer,train_loader)
    test()
