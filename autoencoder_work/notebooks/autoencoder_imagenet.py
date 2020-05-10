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


# In[ ]:


# Check if gpu support is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')


# In[ ]:


# potential good "nominal" classes, appear simpler and more consistent
nominal_classes = ['manhole_cover', 'car_wheel', 'barometer', 'bottlecap', 'lens_cap', 'puck', 'analog_clock', 'wall_clock', 'coffee_mug', 'coffeepot']
nominal_class = nominal_classes[1]
print(f'Running with nominal class: {nominal_class}')


# In[ ]:


im_res=128

model_type = 'AE'

# Autoencoder (AE)
batch_size=256
latent_dim = 8 # Latent Space size

# Variational Autoencoder (VAE)
# TODO

if model_type == 'AE':
    model_name = f'{model_type}_{nominal_class}_latent_dim_{latent_dim}'
    print(f'Compressing from {im_res} x {im_res} to a latent dimension of {latent_dim} x {latent_dim}, ie shinking to {latent_dim}^2/{im_res}^2 = {latent_dim**2 / im_res**2:.2%} of the original size, before expanding back to {im_res} x {im_res}')
elif model_type == 'VAE':
    model_name = f'{model_type}_{nominal_class}_TODO'
    print('TODO')
else:
    raise ValueError(f'Unrecognized model_type = {model_type}')


# In[ ]:


output_path = f'../output/{model_name}'
models_path = f'../models/{model_name}'


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

ds_all_classes = tv.datasets.ImageFolder(root='C:/imagenet/processed_images/train', transform=transform)


# In[ ]:


class_to_idx = OrderedDict({})
for k,v in ds_all_classes.class_to_idx.items():
    class_to_idx[k.lower()] = v
class_to_idx = OrderedDict(sorted(class_to_idx.items(), key=lambda x: x))
idx_to_class = OrderedDict([[v,k] for k,v in class_to_idx.items()])


# In[ ]:


idx_nominal = class_to_idx[nominal_class]
idx_nominal_tensor = torch.tensor(ds_all_classes.targets) == idx_nominal
ds_nominal = torch.utils.data.dataset.Subset(ds_all_classes, np.where(idx_nominal_tensor==1)[0])


# In[ ]:


n_nominal_all = len(ds_nominal.indices)

n_nominal_test = int(0.15*n_nominal_all)
n_nominal_val = int(0.15*n_nominal_all)
n_nominal_train = n_nominal_all - n_nominal_test - n_nominal_val

ds_nominal_test, ds_nominal_val, ds_nominal_train = torch.utils.data.random_split(ds_nominal, [n_nominal_test, n_nominal_val, n_nominal_train])

del ds_nominal; ds_nominal = None;


# In[ ]:


dl_nominal_test = torch.utils.data.DataLoader(ds_nominal_test, batch_size=batch_size, shuffle=False, num_workers=8)
dl_nominal_val = torch.utils.data.DataLoader(ds_nominal_val, batch_size=batch_size, shuffle=False, num_workers=8)
dl_nominal_train = torch.utils.data.DataLoader(ds_nominal_train, batch_size=batch_size, shuffle=False, num_workers=8)


# In[ ]:


# test_mem()


# ***
# # Create the Model

# In[ ]:


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.relu = nn.ReLU()

        # Latent Space size, latent_dim, defined earlier

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


if model_type == 'AE':
    loss_fn = nn.MSELoss()
    loss_fn_no_reduction = nn.MSELoss(reduction='none')

    model = Autoencoder()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
elif model_type == 'VAE':
    print('TODO')


# In[ ]:


# test_mem()


# In[ ]:


dfp_train_results = train_model(dl_nominal_train, dl_nominal_val,
model, optimizer, loss_fn, device,
model_name='autoencoder', models_path=models_path,
max_epochs=100, do_es=True, es_min_val_per_improvement=0.005, es_epochs=10,
do_decay_lr=True, initial_lr=0.001, lr_epoch_period=25, lr_n_period_cap=6,
)


# In[ ]:


write_dfp(dfp_train_results, output_path , 'train_results', tag='',
          target_fixed_cols=['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'delta_per_best', 'saved_model', 'cuda_mem_alloc'],
          sort_by=['epoch'], sort_by_ascending=True)


# ***
# # Eval

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


# ### Load model from disk

# In[ ]:


best_epoch = dfp_train_results.iloc[dfp_train_results['val_loss'].idxmin()]['epoch']
model = Autoencoder()
load_model(model, device, best_epoch, 'autoencoder', models_path)


# In[ ]:


def eval_model(dl, model, loss_fn, loss_fn_no_reduction, device, m_path, fname='im_comp', tag='',
               print_plots=True, n_comps_to_print=20, loss_type='\nLoss is MSE',
               return_loss_stats=True, dl_name=None,
               idx_to_class=idx_to_class, mean_unnormalize=pop_mean, std_unnormalize=pop_std0):
    if not isinstance(loss_fn, nn.modules.loss.MSELoss):
        raise ValueError('Expected loss_fn == nn.MSELoss(), as individual loss annotation on numpy objects uses MSE. Update code and rerun!')

    class_arrays = []
    loss_arrays = []
    model.eval()
    with torch.no_grad():
        if print_plots:
            eval_loss = get_loss(dl, model, loss_fn, device)

        i_comps = 0
        for (images, classes) in tqdm(dl, desc='Minibatch'):
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

            if print_plots:
                def _helper_plot_im_comp(i, _m_path, _fname, _tag='', _ann_add=''):
                    class_name = idx_to_class[classes_np[i]]

                    im_orig = images[i].cpu().numpy()
                    im_pred = outputs[i].cpu().numpy()

                    # compute MSE loss in numpy
                    # this_loss = np.square(np.subtract(im_orig, im_pred)).mean()

                    # get from earlier calculation
                    this_loss = loss_per_image[i]

                    _tag = f'{_tag}_{class_name}{tag}'

                    plot_im_comp(im_orig, im_pred, _m_path, _fname, tag=_tag, inline=False,
                                 ann_text_std_add=f'Loss: {this_loss:.04f}\nMean Loss: {eval_loss:.04f}{loss_type}\n{class_name.title()}{_ann_add}',
                                 mean_unnormalize=mean_unnormalize, std_unnormalize=std_unnormalize,
                                 ann_margin=True, left_right_orig_pred=True,
                                )

                # plot image comparisions, up to n_comps
                i = 0
                n_outputs = len(outputs)
                while i < n_outputs and i_comps < n_comps_to_print:
                    _helper_plot_im_comp(i, m_path, fname, _tag=f'_{i_comps}', _ann_add='')
                    i += 1; i_comps += 1;

                # plot image comparisions for max / min loss in this batch
                i_min = loss_per_image.argmin()
                l_min = loss_per_image[i_min]
                _helper_plot_im_comp(i_min, f'{m_path}/mins', f'min_{l_min:.6f}_fname'.replace('.', '_'), _ann_add=f'\nMin Loss')

                i_max = loss_per_image.argmax()
                l_max = loss_per_image[i_max]
                _helper_plot_im_comp(i_max, f'{m_path}/maxs', f'max_{l_max:.6f}_fname'.replace('.', '_'), _ann_add=f'\nMax Loss')

        if return_loss_stats:
            loss_array = np.concatenate(loss_arrays).ravel()
            class_array = np.concatenate(class_arrays).ravel()
            idxs = natsorted(list(set(class_array)))

            def _get_l_stats(la, name=None):
                return {
                        'name': name, 'l_mean': la.mean(), 'l_stddev': la.std(), 'l_min': la.min(), 'l_max': la.max(),
                        'l_median': np.median(la), 'n_images': la.size
                       }

            results = []
            for idx in tqdm(idxs, desc='Class'):
                this_loss_array = loss_array[np.where(class_array==idx)]
                results.append(_get_l_stats(this_loss_array, name=idx_to_class[idx]))

            if dl_name is not None:
                results.append(_get_l_stats(loss_array, name=dl_name))

            return pd.DataFrame(results)[['name', 'l_mean', 'l_stddev', 'l_min', 'l_max', 'l_median', 'n_images']]


# ### Get Stats on Nominal Class
# Also plot some original / reconstructed image comparisons

# In[ ]:


# nominal
dfp_nominal = eval_model(dl_nominal_val, model, loss_fn, loss_fn_no_reduction, device, f'{output_path}/comps/nominal')
dfp_nominal['nominal'] = 1


# In[ ]:


dfp_nominal


# ### Get Stats on NOT Nominal Classes

# In[ ]:


# not nominal
ds_NOT_nominal = torch.utils.data.dataset.Subset(ds_all_classes, np.where(idx_nominal_tensor!=1)[0])
dl_NOT_nominal = torch.utils.data.DataLoader(ds_NOT_nominal, batch_size=batch_size, shuffle=False, num_workers=8)


# In[ ]:


# compute all not nominal class stats, but don't plot
dfp_NOT_nominal = eval_model(dl_NOT_nominal, model, loss_fn, loss_fn_no_reduction, device, None, print_plots=False, dl_name='not_nominal')
dfp_NOT_nominal['nominal'] = 0


# Combine Results

# In[ ]:


dfp_class_results = pd.concat([dfp_nominal, dfp_NOT_nominal])
dfp_class_results = massage_dfp(dfp_class_results, target_fixed_cols=['nominal', 'name', 'l_mean', 'l_stddev', 'l_min', 'l_max', 'l_median', 'n_images'],
                                sort_by=['nominal', 'l_median'], sort_by_ascending=[False, True])
write_dfp(dfp_class_results, output_path, 'class_results', tag='')


# In[ ]:


dfp_class_results.head(15)


# In[ ]:


dfp_class_results.tail(15)


# ### Plot Image Comparisons for Select NOT Nominal Classes

# In[ ]:


interesting_NOT_nominals = ['analog_clock', 'geyser', 'samoyed', 'scuba_diver', 'comic_book']


# In[ ]:


pbar = tqdm(interesting_NOT_nominals)
for _class in pbar:
    pbar.set_description(f'Processing {_class}')

    idx = class_to_idx[_class]
    this_idx = torch.tensor(ds_all_classes.targets) == idx
    ds = torch.utils.data.dataset.Subset(ds_all_classes, np.where(this_idx==1)[0])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)

    eval_model(dl, model, loss_fn, loss_fn_no_reduction, device, f'{output_path}/comps/{_class}', return_loss_stats=False)


# ***
# # Dev

# In[ ]:


from common_code import *


# In[ ]:


# test_mem()


# In[ ]:





# In[ ]:




