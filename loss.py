import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Kullback-Leibler divergence between VAE latent normal distribution and standard normal distribution.

This is a private function used in the other loss functions. mu and logvar must be pytorch tensors.
"""
def _kl_normal(mu, logvar):
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element).mul_(-0.5)
	return KLD

"""
L1 loss, batch dimension preserved
"""
def _l1_loss(x, y):
	return nn.PairwiseDistance(p=1)(x.view(x.shape[0], -1), y.view(y.shape[0], -1))

"""
L2 loss, batch dimension preserved
"""
def _l2_loss(x, y):
	return nn.PairwiseDistance(p=2)(x.view(x.shape[0], -1), y.view(y.shape[0], -1))

"""
Loss function typically used in variational autoencoders:
	- Image reconstruction loss is capture by binary cross entropy (thus, this only works for grayscale)
	- Difference between latent mean and variance parameters and that of a normal distribution is captured 
	  by KL divergence
"""
def vae_bce_loss(x_orig, x_recon, mu, logvar):
	recon_loss = nn.BCELoss()
	recon_loss.size_average = False
	BCE = recon_loss(x_recon, x_orig)
	KLD = _kl_normal(mu, logvar)
	
	return BCE + KLD

"""
Loss function typically used in variational autoencoders:
	- Image reconstruction loss is captured by mean square error
	- Difference between latent mean and variance parameters and that of a normal distribution is captured 
	  by KL divergence
"""
def vae_mse_loss(x_orig, x_recon, mu, logvar):
	recon_loss = nn.MSELoss(reduction = 'sum')
	MSE = recon_loss(x_recon, x_orig)
	KLD = _kl_normal(mu, logvar)
	
	return MSE + KLD

"""
Loss function typically used in variational autoencoders:
	- Image reconstruction loss is captured by mean absolute error
	- Difference between latent mean and variance parameters and that of a normal distribution is captured 
	  by KL divergence
"""
def vae_mae_loss(x_orig, x_recon, mu, logvar):
	recon_loss = nn.L1Loss(reduction = 'sum')
	MAE = recon_loss(x_recon, x_orig)
	KLD = _kl_normal(mu, logvar)
	
	return MAE + KLD

"""
Loss function typically used in variational autoencoders:
	- Image reconstruction loss is captured by Huber loss
	- Difference between latent mean and variance parameters and that of a normal distribution is captured 
	  by KL divergence
"""
def vae_huber_loss(x_orig, x_recon, mu, logvar):
	recon_loss = nn.SmoothL1Loss(reduction = 'sum')
	huber = recon_loss(x_recon, x_orig)
	KLD = _kl_normal(mu, logvar)
	
	return huber + KLD

"""
Reconstruction loss for a VAE except the cropped out portion the model must regenerate
from scratch is weighted more heavily
"""
def recon_masked_loss(x_orig, x_recon):
	device = 'cuda:0'
	mask = torch.full((1,144,176,64), False, dtype=bool)
	mask[:,:,:,-16:] = True
	mask = mask.to(device)
	weight = 10

	multiplier = torch.ones_like(x_orig)
	multiplier.masked_fill_(mask, weight)
	multiplier = multiplier.to(device)
	loss = nn.L1Loss(reduction = 'none')(x_recon, x_orig)
	loss.mul_(multiplier)

	return loss.sum()



"""
Loss function as implemented in Tudosiu et al. 2020 with a VQ-VAE. 
Utilizes image gradient loss as described in Baur et al. 2019 and KL divergence
"""
def baur_gradient_loss(x_orig, x_recon, mu, logvar):
	
	def gradient(x):
		# borrowed from https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/6
		#
		# x should be of shape (batch size, channels, dim1, dim2, dim3)

		# I have no idea if these are the right axes, but as long as you do the same operation for each axis it don't matter right?
		left = x
		right = F.pad(x, (0, 1, 0, 0, 0, 0))[:, :, :, :, 1:]
		top = x
		bottom = F.pad(x, (0, 0, 0, 1, 0, 0))[:, :, :, 1:, :]
		front = x
		back = F.pad(x, (0, 0, 0, 0, 0, 1))[:, :, 1:, :, :]

		dx, dy, dz = right - left, bottom - top, back - front
		dx[:, :, :, :, -1] = 0
		dy[:, :, :, -1, :] = 0
		dz[:, :, -1, :, :] = 0

		return dx, dy, dz

	# Get spatial gradient
	x_orig_dx, x_orig_dy, x_orig_dz = gradient(x_orig)
	x_recon_dx, x_recon_dy, x_recon_dz = gradient(x_recon)

	l1_loss = _l1_loss(x_orig, x_recon)
	l2_loss = _l2_loss(x_orig, x_recon)
	l1_gdl = _l1_loss(x_orig_dx, x_recon_dx) + _l1_loss(x_orig_dy, x_recon_dy) + _l1_loss(x_orig_dz, x_recon_dz)
	l2_gdl = _l2_loss(x_orig_dx, x_recon_dx) + _l2_loss(x_orig_dy, x_recon_dy) + _l2_loss(x_orig_dz, x_recon_dz)

	total_loss = l1_loss + l2_loss + l1_gdl + l2_gdl

	KLD = _kl_normal(mu, logvar)

	return torch.sum(total_loss) + KLD


