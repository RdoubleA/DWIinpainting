import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

import numpy as np
import time, datetime
import pickle
import os

from Dataset import Dataset
from Models import VAE3D
from utils import *
from loss import *


# ===============================================================
# SPECIFY THESE
# ===============================================================
num_train = 40000
batch_size = 128
crop = True
max_epochs = 30
cpu_batch_load_workers = 8
log_every_num_batches = 40
latent_dim = 512
model_save_path = 'models/expanded_latent_gdl'
input_shape = (1,144,168,48)
loss_function = baur_gradient_loss
# ===============================================================
# end
# ===============================================================


# ===============================================================
# SETUP MODEL AND DATA
# ===============================================================

# Parameters
params = {'batch_size': batch_size,
		  'shuffle': True,
		  'num_workers': cpu_batch_load_workers}

# Datasets
dwi_files_npy = np.loadtxt('X_files.txt', dtype=str)
num_val = dwi_files_npy.shape[0] - num_train

# Generators
training_set = Dataset(dwi_files_npy[:num_train], crop = crop)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(dwi_files_npy[num_train:], crop = crop)
validation_generator = data.DataLoader(validation_set, **params)


model = VAE3D(input_shape, latent_dim, 4, 2, 2)

# Optimization function
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Setup GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print(device)
if use_cuda:
	model.cuda()
print(model)
# ===============================================================
# MAIN SCRIPT
# ===============================================================

history = {'train': [], 'test': []}

# Loop over epochs
for epoch in range(max_epochs):
	t0 = time.time()
	# Training
	train_loss = train(model, optimizer, loss_function, training_generator, epoch, log_every_num_batches = log_every_num_batches)

	# Validation
	test_loss = test(model, loss_function, validation_generator)
	
	savepath = os.path.join(model_save_path, 'Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))
	# Save model
	torch.save(model.state_dict(), savepath)
	
	history['train'].append(train_loss)
	history['test'].append(test_loss)
	
	t_epoch = time.time() - t0
	print('====> Total time elapsed for this epoch: {:s}'.format(str(datetime.timedelta(seconds=int(t_epoch)))))


with open(os.path.join(model_save_path, 'history.p'), 'wb') as f:
	pickle.dump(history, f)