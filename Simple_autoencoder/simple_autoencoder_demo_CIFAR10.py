#!/usr/bin/env python
# coding: utf-8

# Adapted From:  
# [Building Autoencoder in Pytorch - Vipul Vaibhaw](https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c)  

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip');
get_ipython().system('{sys.executable} -m pip install -r ../requirements.txt');


# In[ ]:


import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

# from torchvision.datasets import CIFAR10
# from torch.utils.data import DataLoader


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    # img = img / 2 + 0.5 # unnormalize TODO
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# In[ ]:


# get mean and std deviations per channel for later normalization
# do in minibatches, then take the mean over all the minibatches
# adapted from: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
dataloader_unnormalized = torch.utils.data.DataLoader(
    tv.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=4096, shuffle=False, num_workers=8)

pop_mean = []
pop_std0 = []
# pop_std1 = []
for (images, labels) in tqdm(dataloader_unnormalized, desc='Minibatch'):
    # shape = (minibatch_size, 3, 32, 32)
    numpy_images = images.numpy()

    # shape = (3,)
    batch_mean = np.mean(numpy_images, axis=(0,2,3))
    batch_std0 = np.std(numpy_images, axis=(0,2,3))
    # batch_std1 = np.std(numpy_images, axis=(0,2,3), ddof=1)

    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    # pop_std1.append(batch_std1)

# shape = (num_minibatches, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)
# pop_std1 = np.array(pop_std1).mean(axis=0)


# In[ ]:


# pop_mean
# pop_mean = np.array([0.4916779 , 0.4823491 , 0.44675845], dtype=np.float32)


# In[ ]:


# pop_std0
# pop_std0 = np.array([0.24713232, 0.24348037, 0.26160604], dtype=np.float32)


# In[ ]:


# Load and transform the data
transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize(pop_mean, pop_std0)])

trainset = tv.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
dataloader_train = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=False, num_workers=8)

testset = tv.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
dataloader_test = torch.utils.data.DataLoader(testset, batch_size=4096, shuffle=False, num_workers=8)


# In[ ]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(dataloader_unnormalized)
images, labels = dataiter.next()
imshow(images[0])
# In[ ]:





# In[ ]:


# Create the model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# In[ ]:


# Check if gpu support is available
cuda_avail = torch.cuda.is_available()
print(f'cuda_avail = {cuda_avail}')


# In[ ]:


model = Autoencoder()
if cuda_avail:
    model.cuda()
else:
    print('WARNING Running on CPU!')
    model.cpu()

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
loss_fn = nn.MSELoss()


# In[ ]:


# Create a learning rate adjustment function that divides the learning rate by 10 every epoch_period=30 epochs, up to n_period_cap=6 times
def adjust_learning_rate(epoch, initial_lr=0.001, epoch_period=30, n_period_cap=6):
    exponent = min(n_period_cap, int(np.floor(epoch / epoch_period)))
    lr = initial_lr / pow(10, exponent)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[ ]:


os.makedirs('models', exist_ok=True)

def save_models(epoch):
    torch.save(model.state_dict(), f'models/autoencoder_cifar10model_{epoch}.model')
    print('Checkpoint saved')


# In[ ]:


def test():
    model.eval()
    test_loss = 0.0
    for (images, labels) in dataloader_test:
        if cuda_avail:
            images = Variable(images.cuda())
        else:
            images = Variable(images.cpu())

        # apply model and compute loss using images from the test set
        outputs = model(images)
        loss = loss_fn(outputs, images)
        test_loss += loss.cpu().data.item() * images.size(0)

    # Compute the average loss over all test images
    test_loss = test_loss / len(dataloader_test.dataset)

    return test_loss


# In[ ]:


def train(num_epochs):
    best_loss = None

    # for epoch in tqdm(range(num_epochs), desc='Epochs'):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # for (images, labels) in tqdm(dataloader_train, desc='Minibatch'):
        for (images, labels) in dataloader_train:

            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
            else:
                images = Variable(images.cpu())

            # Clear all accumulated gradients
            optimizer.zero_grad()

            # forward
            outputs = model(images)
            loss = loss_fn(outputs, images)

            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            # compute loss
            train_loss += loss.cpu().data.item() * images.size(0)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all training images
        train_loss = train_loss / len(dataloader_train.dataset)

        # Evaluate on the test set
        test_loss = test()

        # Save the model if the test loss is less than our current best
        if epoch == 0 or test_loss < best_loss:
            save_models(epoch)
            best_loss = test_loss

        # Print the metrics
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')


# In[ ]:


train(50)


# # Other data sets

# In[ ]:


stl10 = tv.datasets.STL10(root='../data', split='train', folds=None, transform=transforms.ToTensor(), download=True)


# In[ ]:


stl10


# In[ ]:


stl10[0][0].size()


# In[ ]:


imshow(stl10[0][0])


# In[ ]:


cifar10 = tv.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())


# In[ ]:


cifar10


# In[ ]:


cifar10[0][0].size()


# In[ ]:


imshow(cifar10[6][0])


# In[ ]:


# imagenet = tv.datasets.ImageNet(root='TODO', split='train', download=True) # Can't download - need to torrent

