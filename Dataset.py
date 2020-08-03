import torch
from torch.utils import data
import numpy as np

"""
PyTorch Dataset generator class to be used in DataLoader
"""
class Dataset(data.Dataset):
    """
    file_list = sequence of filepaths to each sample
    corrupt_prob = probability from 0 to 1 of how likely a loaded image will be artificially cropped
    tanh_norm = if True, normalize to range -1 to 1 instead of 0 to 1. Used for GAN architectures
    num_corrupt = number of axial slices to remove from the top for artificial cropping
    """
    def __init__(self, file_list, corrupt_prob = 0.0, tanh_norm = False, num_corrupt = 8):
        self.inputs = file_list
        self.p_corrupt = corrupt_prob
        self.tanh = tanh_norm
        self.num_corrupt = num_corrupt

    def __len__(self):
        return len(self.inputs)

    """
    Generate one sample of data
    """
    def __getitem__(self, index):
        # Select sample
        sample = self.inputs[index]

        # Use corrupt probability to decide whether to blank top few rows of image or not
        r = np.random.uniform()

        # Load data
        Xout = np.load(sample, allow_pickle = True)

        # Corrupt image by cropping - setting a fixed number of slices in axial direction to 0
        if r < self.p_corrupt:
            Xin = np.array(Xout)
            Xin[:,:,-self.num_corrupt:] = 0
        else:
            Xin = np.array(Xout)
        
        if self.tanh:
            Xout = Xout * 2 - 1
            Xin = Xin * 2 - 1

        # Prep in tensor format
        batch_x = torch.from_numpy(Xin).float()
        batch_x = batch_x.view(1,batch_x.shape[0],batch_x.shape[1],batch_x.shape[2])
        batch_y = torch.from_numpy(Xout).float()
        batch_y = batch_y.view(1,batch_y.shape[0],batch_y.shape[1],batch_y.shape[2])
        
        return batch_x, batch_y