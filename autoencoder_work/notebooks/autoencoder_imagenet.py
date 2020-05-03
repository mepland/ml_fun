#!/usr/bin/env python
# coding: utf-8

# Adapted From:  
# [Building Autoencoder in Pytorch - Vipul Vaibhaw](https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c)  
# [Variational AutoEncoders for new fruits with Keras and Pytorch - Thomas Dehaene](https://becominghuman.ai/variational-autoencoders-for-new-fruits-with-keras-and-pytorch-6d0cfc4eeabd)  
import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip');
get_ipython().system('{sys.executable} -m pip install -r ../requirements.txt');
# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys, os
sys.path.append(os.path.expanduser('~/ml_fun/'))
from common_code import *
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import mean_squared_error

output_path = '../output'
models_path = '../models'


# In[ ]:


# Check if gpu support is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')


# In[ ]:


batch_size=256
im_res=128


# In[ ]:


# test_mem()


# ***
# ### Compute Normalization Factors
dl_unnormalized = torch.utils.data.DataLoader(
    tv.datasets.ImageFolder(root='C:/imagenet/processed_images/train', transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False, num_workers=8
)

pop_mean, pop_std0 = compute_channel_norms(dl_unnormalized)

print(f'pop_mean = {pop_mean}')
print(f'pop_std0 = {pop_std0}')
# In[ ]:


# use normalization results computed earlier
pop_mean = np.array([0.48399296, 0.45583892, 0.41094956])
pop_std0 = np.array([0.27657014, 0.27107376, 0.28344524])


# ***
# # Load and manipulate data

# In[ ]:


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(pop_mean, pop_std0)])

# ds_all_classes = tv.datasets.ImageFolder(root='C:/imagenet/processed_images/train', transform=transform)
ds_all_classes = tv.datasets.ImageFolder(root='C:/imagenet/processed_images/train_subset_of_classes', transform=transform)


# In[ ]:


class_to_idx = OrderedDict({})
for k,v in ds_all_classes.class_to_idx.items():
    class_to_idx[k.lower()] = v
class_to_idx = OrderedDict(sorted(class_to_idx.items(), key=lambda x: x))
idx_to_class = OrderedDict([[v,k] for k,v in class_to_idx.items()])


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


# In[ ]:


n_dogs_all = len(ds_dogs.indices)

n_dogs_test = int(0.15*n_dogs_all)
n_dogs_val = int(0.15*n_dogs_all)
n_dogs_train = n_dogs_all - n_dogs_test - n_dogs_val

ds_dogs_test, ds_dogs_val, ds_dogs_train = torch.utils.data.random_split(ds_dogs, [n_dogs_test, n_dogs_val, n_dogs_train])

del ds_dogs; ds_dogs = None;


# In[ ]:


dl_dogs_test = torch.utils.data.DataLoader(ds_dogs_test, batch_size=batch_size, shuffle=False, num_workers=8)
dl_dogs_val = torch.utils.data.DataLoader(ds_dogs_val, batch_size=batch_size, shuffle=False, num_workers=8)
dl_dogs_train = torch.utils.data.DataLoader(ds_dogs_train, batch_size=batch_size, shuffle=False, num_workers=8)


# In[ ]:


# test_mem()


# ***
# # Create the Model

# In[ ]:


# Create the model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.relu = nn.ReLU()

        # Latent Space size
        latent_dim = 8

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(latent_dim)

        # Decoder
        self.conv5 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
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


loss_fn = nn.MSELoss()
loss_fn_no_reduction = nn.MSELoss(reduction='none')


# In[ ]:


model = Autoencoder()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)


# In[ ]:


# test_mem()


# In[ ]:


dfp_train_results = train_model(dl_dogs_train, dl_dogs_val,
model, optimizer, loss_fn, device,
model_name='autoencoder', models_path=models_path,
max_epochs=100, do_es=True, es_min_val_per_improvement=0.005, es_epochs=10,
do_decay_lr=True, initial_lr=0.001, lr_epoch_period=30, lr_n_period_cap=6,
)


# In[ ]:


write_dfp(dfp_train_results, output_path , 'train_results', tag='',
          target_fixed_cols=['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'delta_per_best', 'saved_model', 'cuda_mem_alloc'],
          sort_by=['epoch'], sort_by_ascending=True)


# In[ ]:


dfp_train_results = load_dfp(output_path, 'train_results', tag='', cols_bool=['saved_model'],
                             cols_float=['train_loss','val_loss','best_val_loss','delta_per_best'])


# In[ ]:


# dfp_train_results


# In[ ]:


plot_loss_vs_epoch(dfp_train_results, output_path, fname='loss_vs_epoch', tag='', inline=False,
                   ann_text_std_add=None,
                   y_axis_params={'log': True},
                   loss_cols=['train_loss', 'val_loss'],
                  )


# ***
# # Eval

# In[ ]:


model = Autoencoder()
load_model(model, device, 40, 'autoencoder', models_path)


# In[ ]:


def eval_model(dl, model, loss_fn, loss_fn_no_reduction, device, m_path, fname='im_comp', tag='', n_comps_to_print=50, loss_type='\nLoss is MSE',
               return_loss_stats=True, report_loss_per_class=True, dl_name=None,
               idx_to_class=idx_to_class, mean_unnormalize=pop_mean, std_unnormalize=pop_std0):
    if not isinstance(loss_fn, nn.modules.loss.MSELoss):
        raise ValueError('Expected loss_fn == nn.MSELoss(), as individual loss annotation on numpy objects uses MSE. Update code and rerun!')

    class_arrays = []
    loss_arrays = []
    model.eval()
    with torch.no_grad():
        eval_loss = get_loss(dl, model, loss_fn, device)

        i_comps = 0
        for (images, classes) in dl:
            # move labels to cpu
            classes_np = classes.numpy()
            class_arrays.append(classes_np)

            # move data to device
            images = images.to(device)

            # evaluate with model
            outputs = model(images)

            loss_per_pixel = loss_fn_no_reduction(outputs, images).cpu().numpy()
            loss_per_image = np.reshape(loss_per_pixel, (loss_per_pixel.shape[0], -1)).mean(axis=1)
            loss_arrays.append(loss_per_image)

            # plot image comparisions, up to n_comps
            i = 0
            n_outputs = len(outputs)
            while i < n_outputs and i_comps < n_comps_to_print:
                idx = classes_np[i]
                class_name = idx_to_class[idx]

                im_orig = images[i].cpu().numpy()
                im_pred = outputs[i].cpu().numpy()

                # compute MSE loss in numpy
                # this_loss = np.square(np.subtract(im_orig, im_pred)).mean()

                # get from earlier calculation
                this_loss = loss_per_image[i]

                plot_im_comp(im_orig, im_pred, m_path, fname, tag=f'_{i_comps}_{class_name}{tag}', inline=False,
                             ann_text_std_add=f'Loss: {this_loss:.04f}\nMean Loss: {eval_loss:.04f}{loss_type}\n{class_name.title()}',
                             mean_unnormalize=mean_unnormalize, std_unnormalize=std_unnormalize,
                             ann_margin=True, left_right_orig_pred=True,
                            )

                i += 1; i_comps += 1;

    if return_loss_stats:
        loss_array = np.concatenate(loss_arrays).ravel()

        def _get_l_stats(la, name=None):
            l_mean = la.mean()
            l_stddev = la.std()
            l_min = la.min()
            l_max = la.max()
            l_median = np.median(la)

            return {'name': name, 'l_mean': l_mean, 'l_stddev': l_stddev, 'l_min': l_min, 'l_max': l_max, 'l_median': l_median}

        results = []
        if report_loss_per_class:
            class_array = np.concatenate(class_arrays).ravel()
            idxs = natsorted(list(set(class_array)))
            for idx in idxs:
                this_loss_array = loss_array[np.where(class_array==idx)]
                class_name = idx_to_class[idx]
                results.append(_get_l_stats(this_loss_array, name=class_name))

        if dl_name is not None:
            results.append(_get_l_stats(loss_array, name=dl_name))

        return pd.DataFrame(results)[['name', 'l_mean', 'l_stddev', 'l_min', 'l_max', 'l_median']]


# In[ ]:


# dogs
dfp_dogs = eval_model(dl_dogs_val, model, loss_fn, loss_fn_no_reduction, device, f'{output_path}/comps/dogs', dl_name='dogs')
dfp_dogs['dogs'] = 1


# In[ ]:


dfp_dogs


# In[ ]:


# not dogs, individually
for _class,idx in class_to_idx.items():
    if _class in possible_dog_classes:
        continue
    print(f'Processing {_class}')

    this_idx = torch.tensor(ds_all_classes.targets) == idx
    ds = torch.utils.data.dataset.Subset(ds_all_classes, np.where(this_idx==1)[0])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)

    eval_model(dl, model, loss_fn, loss_fn_no_reduction, device, f'{output_path}/comps/{_class}', return_loss_stats=False)


# In[ ]:


# not dogs, together
ds_NOT_dogs = torch.utils.data.dataset.Subset(ds_all_classes, np.where(idx_dogs!=1)[0])
dl_NOT_dogs = torch.utils.data.DataLoader(ds_NOT_dogs, batch_size=batch_size, shuffle=False, num_workers=8)


# In[ ]:


dfp_NOT_dogs = eval_model(dl_NOT_dogs, model, loss_fn, loss_fn_no_reduction, device, None, n_comps_to_print=0, dl_name='not_dogs')
dfp_NOT_dogs['dogs'] = 0


# In[ ]:


dfp_NOT_dogs


# In[ ]:





# In[ ]:


dfp_class_results = pd.concat([dfp_dogs, dfp_NOT_dogs])


# In[ ]:


write_dfp(dfp_class_results, output_path, 'class_results', tag='', target_fixed_cols=['name', 'l_mean', 'l_stddev', 'l_min', 'l_max', 'l_median'],
          sort_by=None, sort_by_ascending=None,
         )


# ***
# # Dev

# In[ ]:


from common_code import *


# In[ ]:


# test_mem()


# In[ ]:





# In[ ]:




