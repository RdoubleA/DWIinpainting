import torch
import torch.nn as nn
import time, datetime
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_anat
from skimage.metrics import structural_similarity

"""
Trains a VAE model with a given optimizer, loss function, training data

Arguments
    - model: torch.nn.module, in this case it's the VAE3D class from the Models.py file
    - optimizer: one of the pytorch optimizers
    - loss_function: pointer to a function to compute model loss, should take in arguments 
      (original image, reconstructed image, latent mean, latent logvariance)
    - training_generator: pytorch DataLoader object that is a generator to yield batches of training data
    - epoch: integer specifying which epoch we're on, zero indexed. sole purpose is for printing progress
    - log_every_num_batches: integere specify after how many batches do we print progress
"""
def train(model, optimizer, loss_function, training_generator, epoch, log_every_num_batches = 40):
    print('====> Begin epoch {}'.format(epoch+1))
    print()
    # Setup GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
    model.to(device)

    t0 = time.time()
    model.train()
    train_loss = 0
    batch_id = 1
    batch_size = training_generator.batch_size
    for batch_in, batch_out in training_generator:
        batch_run_time = time.time()
        # Transfer to GPU
        batch_in, batch_out = batch_in.to(device), batch_out.to(device)
        
        # Clear optimizer gradients
        optimizer.zero_grad()
        # Forward pass through the model
        outputs = model(batch_in)
        
        # Calculate loss
        recon_loss = loss_function(outputs['x_out'], batch_out)
        if 'vq_loss' in outputs:
            loss = recon_loss + outputs['vq_loss']
        else:
            loss = recon_loss

        # Back propagate
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        batch_run_time = time.time() - batch_run_time
        et = batch_run_time * (len(training_generator) - batch_id) * 3
        # Print progress
        if batch_id % log_every_num_batches == 0 or batch_id == 1:
            print('Train Epoch: {:d} [{:d}/{:d} ({:.0f}%)]\tLoss: {:.6f}\tET - {:s}'.format(
                    epoch+1, batch_id, len(training_generator),
                    100. * batch_id / len(training_generator),
                    loss.item() / len(batch_in), str(datetime.timedelta(seconds=int(et)))))
        batch_id += 1
        
    t_epoch = time.time() - t0
    train_loss /= len(training_generator) * batch_size
    print()
    print('====> Epoch: {} Average loss: {:.4f}\tTime elapsed: {:s}'.format(
          epoch+1, train_loss, str(datetime.timedelta(seconds=int(t_epoch)))))
    return train_loss

"""
Evaluates a VAE model with a given validation/test set

Arguments
    - model: torch.nn.module, in this case it's the VAE3D class from the Models.py file
    - loss_function: pointer to a function to compute model loss, should take in arguments 
      (original image, reconstructed image, latent mean, latent logvariance)
    - validation_generator: pytorch DataLoader object that is a generator to yield batches of test data
"""
def test(model, loss_function, validation_generator):
    # Setup GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    test_loss = 0
    t0 = time.time()
    batch_size = validation_generator.batch_size
    with torch.no_grad():
        for batch_in, batch_out in validation_generator:
            # Transfer to GPU
            batch_in, batch_out = batch_in.to(device), batch_out.to(device)
            # Forward pass through the model
            outputs = model(batch_in)
            
            # Calculate loss
            recon_loss = loss_function(outputs['x_out'], batch_out)
            if 'vq_loss' in outputs:
                loss = recon_loss + outputs['vq_loss']
            else:
                loss = recon_loss
            test_loss += loss.item()
    
    t_epoch = time.time() - t0
    test_loss /= len(validation_generator) * batch_size
    print('====> Test set loss: {:.4f}\tTime elapsed: {:s}'.format(
        test_loss, str(datetime.timedelta(seconds=int(t_epoch)))))
    print()
    return test_loss

"""
Plot the training and validation history of a model
"""
def plot_loss(history, xmin = 0, xmax = 30, ymin = 0, ymax = 5000):
    plt.close('all')
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    ax.plot(history['train'], '-d', color='purple')
    ax.plot(history['test'], '-d', color='orange')
    ax.legend(['train loss', 'validation loss'])
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_title('VAE training on cropped DTI')
    plt.show()

"""
Test the model on validation data set and plot generated images
"""
def test_model(model, validation_generator, num_plot = 6):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    plot_batch_in, plot_batch_out = next(iter(validation_generator))
    plot_batch_in, plot_batch_out = plot_batch_in.to(device), plot_batch_out.to(device)
    model.eval()
    model.to(device)
    outputs = model(plot_batch_in)
    if ("rec") in outputs:
        plot_recon = outputs[("rec")]
    else:
        plot_recon = outputs['x_out']
    x_recon = plot_recon.cpu().detach().numpy().squeeze()
    x_corr = plot_batch_in.cpu().detach().numpy().squeeze()
    x_clean = plot_batch_out.cpu().detach().numpy().squeeze()

    ssim_orig2recon = [structural_similarity(x_clean[i][:,:,-8:], x_recon[i][:,:,-8:], data_range = 1) for i in range(x_recon.shape[0])]

    for i in range(num_plot):
        plt.close('all')
        fig,ax = plt.subplots(3,1,figsize = (16,8))
        # Convert numpy array to nifti
        dwi_clean = nib.Nifti1Image(x_clean[i], np.eye(4))
        dwi_corr = nib.Nifti1Image(x_corr[i], np.eye(4))
        dwi_recon = nib.Nifti1Image(x_recon[i], np.eye(4))
        # Plot original
        plot_anat(dwi_clean, axes = ax[0], vmin = -0.1, vmax = 0.3)
        # Plot corrupt
        plot_anat(dwi_corr, axes = ax[1], vmin = -0.1, vmax = 0.3)
        # Plot reconstructed
        plot_anat(dwi_recon, axes = ax[2], vmin = -0.1, vmax = 0.3)
        
        ax[0].set_title('original')
        ax[1].set_title('cropped')
        ax[2].set_title('reconstructed, SSIM = {:.6f}'.format(ssim_orig2recon[i]))
        plt.show()
        
    print('Average SSIM: %.6f' % np.mean(ssim_orig2recon))
    print('STD SSIM: %.6f' % np.std(ssim_orig2recon))

    return np.mean(ssim_orig2recon), np.std(ssim_orig2recon)
