import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image

########################################################
# load package wide variables
from .configs import *
from .pandas_functions import *

########################################################
def test_mem():
	cuda_mem_alloc = torch.cuda.memory_allocated() # bytes
	print(f'CUDA memory allocated: {humanize.naturalsize(cuda_mem_alloc)}')
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
				print(f'type: {type(obj)}, dimensional size: {obj.size()}') # , memory size: {humanize.naturalsize(sys.getsizeof(obj))}') - always 72...
		except:
			pass

########################################################
# get mean and std deviations per channel for later normalization
# do in minibatches, then take the mean over all the minibatches
# adapted from: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
def compute_channel_norms(dl, do_std1=False):
	pop_mean = []
	pop_std0 = []
	pop_std1 = []
	for (images, _) in tqdm(dl, desc='Minibatch'):
		# shape = (batch_size, 3, im_res, im_res)
		numpy_images = images.numpy()

		# shape = (3,)
		batch_mean = np.mean(numpy_images, axis=(0,2,3))
		batch_std0 = np.std(numpy_images, axis=(0,2,3))
		if do_std1:
			batch_std1 = np.std(numpy_images, axis=(0,2,3), ddof=1)

		pop_mean.append(batch_mean)
		pop_std0.append(batch_std0)
		if do_std1:
			pop_std1.append(batch_std1)

	# shape = (num_minibatches, 3) -> (mean across 0th axis) -> shape (3,)
	pop_mean = np.array(pop_mean).mean(axis=0)
	pop_std0 = np.array(pop_std0).mean(axis=0)
	if do_std1:
		pop_std1 = np.array(pop_std1).mean(axis=0)

	if do_std1:
		return pop_mean, pop_std0, pop_std1
	else:
		return pop_mean, pop_std0

########################################################
def save_model(model, epoch, model_name, models_path):
	os.makedirs(models_path, exist_ok=True)
	torch.save(model.state_dict(), os.path.join(models_path, f'{model_name}_{epoch}.model'))

########################################################
def load_model(model, device, epoch, model_name, models_path):
	# model is the base class of the model you want to load, will have loaded model in the end
	model.to(device)
	model.load_state_dict(torch.load(os.path.join(models_path, f'{model_name}_{epoch}.model')))

########################################################
# learning rate adjustment function that divides the learning rate by 10 every lr_epoch_period=30 epochs, up to lr_n_period_cap=6 times
def decay_lr(optimizer, epoch, initial_lr=0.001, lr_epoch_period=30, lr_n_period_cap=6):
	exponent = min(lr_n_period_cap, int(np.floor(epoch / lr_epoch_period)))
	lr = initial_lr / pow(10, exponent)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

########################################################
def get_loss(dl, model, loss_fn, device):
	model.eval()
	total_loss = 0.0
	for (images, _) in dl:
		images = images.to(device)

		# apply model and compute loss using images from the dataloader dl
		outputs = model(images)
		loss = loss_fn(outputs, images)
		total_loss += loss.cpu().data.item() * images.size(0) # mean loss of batch * number of images in batch = sum of per image losses

	# Compute the mean loss over all images
	mean_loss = total_loss / len(dl.dataset) # sum of per image losses / n images

	return mean_loss

########################################################
def train_model(dl_train, dl_val,
model, optimizer, loss_fn, device,
model_name, models_path,
max_epochs, do_es=True, es_min_val_per_improvement=0.005, es_epochs=10,
do_decay_lr=True, initial_lr=0.001, lr_epoch_period=30, lr_n_period_cap=6,
print_CUDA_MEM=False,
):
	float_fmt='.9f'

	best_val_loss = None
	training_results = []
	all_val_losses = []
	# for epoch in tqdm(range(max_epochs), desc='Epoch'):

	epoch_pbar = tqdm(total=max_epochs, desc='Epoch', position=0)
	for epoch in range(max_epochs):
		model.train()
		# for (images, _) in tqdm(dl_train, desc='Minibatch', position=1): # works, but keeps repeating this pbar
		for (images, _) in dl_train:
			# Move images to gpu if available
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

		if do_decay_lr:
			decay_lr(optimizer, epoch, initial_lr, lr_epoch_period, lr_n_period_cap)

		# Compute the train_loss here via get_loss, with its own loop through the data, for an apples-to-apples comparison at the end of the epoch's training
		train_loss = get_loss(dl_train, model, loss_fn, device)

		# Evaluate on the val set
		val_loss = get_loss(dl_val, model, loss_fn, device)

		# Start printing epoch_message
		delta_per_best = 0
		if epoch != 0:
			delta_per_best = (val_loss-best_val_loss) / best_val_loss
		epoch_message = f'Epoch {epoch:4d}, Train Loss: {train_loss:{float_fmt}}, Val Loss: {val_loss:{float_fmt}}, Delta Best {delta_per_best:8.3%}'

		# Save the model if the val loss is less than our current best
		saved_model = False
		if epoch == 0 or val_loss < best_val_loss:
			save_model(model, epoch, model_name, models_path)
			best_val_loss = val_loss
			saved_model = True

		# Finish epoch_message
		cuda_mem_alloc = None
		if str(device) == 'cuda':
			cuda_mem_alloc = torch.cuda.memory_allocated() # bytes
			if print_CUDA_MEM:
				epoch_message = f'{epoch_message}, CUDA MEM: {humanize.naturalsize(cuda_mem_alloc)}'

		if saved_model:
			epoch_message = f'{epoch_message}, Model Saved!'
		epoch_pbar.write(epoch_message)

		# save the metrics
		training_results.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'best_val_loss': best_val_loss, 'delta_per_best': delta_per_best, 'saved_model': saved_model, 'cuda_mem_alloc': cuda_mem_alloc})
		all_val_losses.append(val_loss)

		# check for early stopping
		if do_es and epoch > es_epochs:
			ref_val_loss = all_val_losses[-es_epochs]
			per_changes = [(ref_val_loss - past_val_loss) / ref_val_loss for past_val_loss in all_val_losses[-es_epochs:]]
			execute_es = True
			for per_change in per_changes:
				if per_change > es_min_val_per_improvement:
					execute_es = False
					break
			if execute_es:
				# print message and early stop
				epoch_pbar.write(f'\nOver the past {es_epochs} epochs the val loss did not improve by at least {es_min_val_per_improvement:%}, stopping early!')
				# these messages are for debugging
				# epoch_pbar.write(f'ref_val_loss: {ref_val_loss:{float_fmt}}')
				# epoch_pbar.write(f"per_changes: {', '.join([f'{per:8.3%}' for per in per_changes])}")
				break

		# end of epoch loop, update pbar
		epoch_pbar.update(1)

	# training complete, wrap it up
	dfp_train_results = create_dfp(training_results, target_fixed_cols=['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'delta_per_best', 'saved_model', 'cuda_mem_alloc'], sort_by=['epoch'], sort_by_ascending=[True])
	return dfp_train_results
