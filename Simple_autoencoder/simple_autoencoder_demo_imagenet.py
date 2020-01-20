#!/usr/bin/env python
# coding: utf-8

# Adapted From:  
# [Building Autoencoder in Pytorch - Vipul Vaibhaw](https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c)  
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip');
get_ipython().system('{sys.executable} -m pip install -r ../requirements.txt');
# In[ ]:


import os
import gc
import sys
import numpy as np
from collections import OrderedDict
from natsort import natsorted
import humanize
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')


# In[ ]:


batch_size=256
im_res=128


# In[ ]:


def test_mem():
    cuda_mem_alloc = torch.cuda.memory_allocated() # bytes
    print(f'CUDA memory allocated: {humanize.naturalsize(cuda_mem_alloc)}')
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(f'type: {type(obj)}, dimensional size: {obj.size()}') # , memory size: {humanize.naturalsize(sys.getsizeof(obj))}') - always 72...
        except:
            pass


# In[ ]:


# test_mem()


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

# del idx_dogs
# idx_dogs = None


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


# ***
# # Create the Model

# In[ ]:


# Create the model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.relu = nn.ReLU()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        # Decoder
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))

        return conv4

    def decode(self, z):
        conv5 = self.relu(self.bn5(self.conv5(z)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))

        return self.conv8(conv7).view(-1, 3, im_res, im_res)

    def forward(self, x):
        return self.decode(self.encode(x))


# In[ ]:


model = Autoencoder()
model.to(device)

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

def save_model(epoch, model_name='autoencoder'):
    torch.save(model.state_dict(), f'models/{model_name}_{epoch}.model')
    print('Checkpoint saved')


# In[ ]:


def load_model(epoch, model_name='autoencoder'):
    model = Autoencoder()
    model.to(device)
    model.load_state_dict(torch.load(f'models/{model_name}_{epoch}.model'))
    return model


# In[ ]:


def get_val_loss():
    model.eval()
    val_loss = 0.0
    for (images, labels) in dl_dogs_val:
        images = images.to(device)

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
            images = images.to(device)

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
            save_model(epoch)
            best_loss = val_loss

        # Print the metrics
        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')


# In[ ]:


train(100)


# ***
# # Dev

# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

def imshow(im, mean=pop_mean, std=pop_std0):
    if isinstance(im, torch.Tensor):
        im = im.numpy()
    # unnormalize
    if std is not None and mean is not None:
        im_unnorm = np.zeros(im.shape)
        for channel in range(im.shape[0]):
            im_unnorm[channel] = std[channel]*im[channel] + mean[channel]
        im = im_unnorm
        del im_unnorm

    # transpose from (channels, im_res, im_res) to (im_res, im_res, channels) for imshow plotting
    im = np.transpose(im, (1, 2, 0))
    
    plt.imshow(im)
    plt.show()


# In[ ]:





# In[ ]:


images, labels = iter(dl_dogs_val).next()


# In[ ]:


imshow(images[10])


# In[ ]:





# In[ ]:


model = load_model(91)
# model = load_model(1)


# In[ ]:


outputs = model(images.to(device))
outputs_cpu = outputs.data.cpu().numpy()


# In[ ]:


imshow(outputs_cpu[10])


# In[ ]:





# In[ ]:


ds_NOT_dogs = torch.utils.data.dataset.Subset(ds_all_classes, np.where(idx_dogs!=1)[0])
dl_NOT_dogs = torch.utils.data.DataLoader(ds_NOT_dogs, batch_size=100, shuffle=False, num_workers=8)


# In[ ]:





# In[ ]:


images, labels = iter(dl_NOT_dogs).next()


# In[ ]:


imshow(images[10])


# In[ ]:


outputs = model(images.to(device))
outputs_cpu = outputs.data.cpu().numpy()


# In[ ]:


imshow(outputs_cpu[10])

