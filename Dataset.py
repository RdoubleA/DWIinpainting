import torch
from torch.utils import data
import numpy as np

"""
PyTorch Dataset generator class to be used in DataLoader

This one is specifically tailored for unsupervised models
"""
class Dataset(data.Dataset):
    """
    file_list = sequence of filepaths to each sample
    """
    def __init__(self, file_list, corrupt_prob = 0.0, crop = False, pad = False, tanh_norm = False, num_corrupt = 16):
        self.inputs = file_list
        self.crop = crop
        self.pad = pad
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
        #print('loading single sample')
        Xout = np.load(sample)
        # Padding is requires if we are to use more than three downsampling layers (divide each dimension by 2 more than twice)
        if self.pad:
            Xout = np.pad(Xout, pad_width=((0,0),(4,4),(0,0)))
        if self.crop:
            Xout = Xout[:,:,-64:]
        # else:
        #     # Zero pad to make z-dimension divisible by 2, will make convolutional layers work better
        #     Xout = np.pad(Xout,pad_width=((0,0),(0,0),(0,1)))

        if r < self.p_corrupt:
            Xin = np.array(Xout)
            Xin[:,:,-self.num_corrupt:] = 0
        else:
            Xin = np.array(Xout)
        
        if self.tanh:
            Xout = Xout * 2 - 1
            Xin = Xin * 2 - 1

        batch_x = torch.from_numpy(Xin).float()
        batch_x = batch_x.view(1,batch_x.shape[0],batch_x.shape[1],batch_x.shape[2])
        batch_y = torch.from_numpy(Xout).float()
        batch_y = batch_y.view(1,batch_y.shape[0],batch_y.shape[1],batch_y.shape[2])
        
        return batch_x, batch_y