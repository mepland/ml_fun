#!/usr/bin/env python
# coding: utf-8

# Adapted From:  
# [Building Autoencoder in Pytorch - Vipul Vaibhaw](https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c)  
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip');
get_ipython().system('{sys.executable} -m pip install -r ../requirements.txt');
# In[ ]:


import os
import numpy as np
from collections import OrderedDict
from natsort import natsorted
from tqdm import tqdm

import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms
from torchvision.utils import save_image

# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


# In[ ]:


# Check if gpu support is available
cuda_avail = torch.cuda.is_available()
print(f'cuda_avail = {cuda_avail}')


# In[ ]:


batch_size=1024
# im_res=128


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    # img = img / 2 + 0.5 # unnormalize TODO
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ***
# ### Compute Normalization Factors
# get mean and std deviations per channel for later normalization
# do in minibatches, then take the mean over all the minibatches
# adapted from: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7

dl_unnormalized = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(root='C:/imagenet/processed_images/train', transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False, num_workers=8
)

pop_mean = []
pop_std0 = []
# pop_std1 = []
for (images, labels) in tqdm(dl_unnormalized, desc='Minibatch'):
    # shape = (batch_size, 3, im_res, im_res)
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

print(f'pop_mean = {pop_mean}')
print(f'pop_std0 = {pop_std0}')
# print(f'pop_std1 = {pop_std1}')
# In[ ]:


# use normalization results computed earlier
pop_mean = np.array([0.48399296, 0.45583892, 0.41094956])
pop_std0 = np.array([0.27657014, 0.27107376, 0.28344524])
# pop_std1 = np.array()


# ***
# # Load and manipulate data

# In[ ]:


transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize(pop_mean, pop_std0)])

ds_all_classes = tv.datasets.ImageFolder(root='C:/imagenet/processed_images/train', transform=transform)


# In[ ]:


class_to_idx = OrderedDict({})
for k,v in ds_all_classes.class_to_idx.items():
    class_to_idx[k.lower()] = v
class_to_idx = OrderedDict(sorted(class_to_idx.items(), key=lambda x: x))


# In[ ]:


# From https://www.kaggle.com/c/dog-breed-identification/data, plus a few extra
possible_dog_classes = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier', 'dalmatian', 'coyote', 'timber_wolf', 'white_wolf',]


# In[ ]:


imagenet_dog_classes = natsorted(list(set(class_to_idx.keys()).intersection(set(possible_dog_classes))))

imagenet_dog_classes_idx = []
for c in imagenet_dog_classes:
    imagenet_dog_classes_idx.append(class_to_idx[c])


# In[ ]:


for i,class_idx in enumerate(imagenet_dog_classes_idx):
    if i == 0:
        idx_dogs = torch.tensor(ds_all_classes.targets) == class_idx
    else:
        idx_dogs += torch.tensor(ds_all_classes.targets) == class_idx


# In[ ]:


ds_dogs = torch.utils.data.dataset.Subset(ds_all_classes, np.where(idx_dogs==1)[0])

del ds_all_classes
ds_all_classes = None


# In[ ]:


n_dogs_all = len(ds_dogs.indices)

n_dogs_test = int(0.15*n_dogs_all)
n_dogs_val = int(0.15*n_dogs_all)
n_dogs_train = n_dogs_all - n_dogs_test - n_dogs_val

ds_dogs_test, ds_dogs_val, ds_dogs_train = torch.utils.data.random_split(ds_dogs, [n_dogs_test, n_dogs_val, n_dogs_train])

del ds_dogs
ds_dogs = None


# In[ ]:


dl_dogs_test = torch.utils.data.DataLoader(ds_dogs_test, batch_size=batch_size, shuffle=False, num_workers=8)
dl_dogs_val = torch.utils.data.DataLoader(ds_dogs_val, batch_size=batch_size, shuffle=False, num_workers=8)
dl_dogs_train = torch.utils.data.DataLoader(ds_dogs_train, batch_size=batch_size, shuffle=False, num_workers=8)

dataiter = iter(dataloader_train_dogs)
images, labels = dataiter.next()
imshow(images[110])
# ***
# # Create the Model

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
    torch.save(model.state_dict(), f'models/autoencoder_{epoch}.model')
    print('Checkpoint saved')


# In[ ]:


def get_val_loss():
    model.eval()
    val_loss = 0.0
    for (images, labels) in dl_dogs_val:
        if cuda_avail:
            images = Variable(images.cuda())
        else:
            images = Variable(images.cpu())

        # apply model and compute loss using images from the val set
        outputs = model(images)
        loss = loss_fn(outputs, images)
        val_loss += loss.cpu().data.item() * images.size(0)

    # Compute the average loss over all val images
    # val_loss = val_loss / len(dl_dogs_val.dataset)
    val_loss = val_loss / n_dogs_val

    return val_loss


# In[ ]:


def train(num_epochs):
    best_loss = None

    # for epoch in tqdm(range(num_epochs), desc='Epochs'):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # for (images, labels) in tqdm(dataloader_train, desc='Minibatch'):
        for (images, labels) in dl_dogs_train:

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
        # train_loss = train_loss / len(dl_dogs_train.dataset)
        train_loss = train_loss / n_dogs_train

        # Evaluate on the val set
        val_loss = get_val_loss()

        # Save the model if the val loss is less than our current best
        if epoch == 0 or val_loss < best_loss:
            save_models(epoch)
            best_loss = val_loss

        # Print the metrics
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')


# In[ ]:


train(500)


# In[ ]:




