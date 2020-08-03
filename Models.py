import torch
import torch.nn as nn
from model_parts import *


class VQVAE3D(nn.Module):
    """
    VQVAE and U-VQVAE models

    Replace the latent layer in a variational autoencoder with vector quantization as described in Oord et al. 2017
    Also include skip connections

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    embedding_dim = dimensionality of vectors in the embedding space
    num_embeddings = number of vectors in the embedding space
    skip_connections = if True, the model is U-VQVAE. if False, the model is VQVAE.
    batchnorm = if True, include batchnorm layers after every convolutional layer

    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 512, skip_connections = True, batchnorm = False):
        super().__init__()
        self.skip = skip_connections
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = skip_connections, batchnorm = batchnorm)
        self.quantization = VectorQuantizerEMA(num_embeddings = num_embeddings, embedding_dim = embedding_dim)
        # If skip connections enabled, use a different class for decoder that uses skip connections
        if skip_connections:
            self.decoder = VQDecoder_skip(num_channels, num_filters, embedding_dim, batchnorm)
        else:
            self.decoder = VQDecoder(num_channels, num_filters, embedding_dim, batchnorm)

    def forward(self, x):
        if self.skip:
            x1, x2, x3, ze = self.encoder(x)
            loss, zq, perplexity, _ = self.quantization(ze)
            x_recon = self.decoder(zq, x1, x2, x3)
        else:
            ze = self.encoder(x)
            loss, zq, perplexity, _ = self.quantization(ze)
            x_recon = self.decoder(zq)

        outputs = {'x_out': x_recon,
                   'vq_loss': loss}

        return outputs


class UNet(nn.Module):
    """
    U-shaped VQVAE architecture with skip connections from encoder layers to decoder, 
    similar to original U-net by Ronneberger et al. 2015, but not the exact same.

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first Down layer, this doubles after every Down layer    
    bilinear = if True, use bilinear interpolation to upsample in the decoder, which does not use learned weights
               if False, use transpose convolutions with learned weights, which is more memory intensive

    """
    def __init__(self, num_channels, num_filters = 4, bilinear = True):
        super().__init__()
        # Encoding
        self.inconv = DoubleConv(num_channels, num_filters)
        self.down1 = Down(num_filters, num_filters * 2)
        self.down2 = Down(num_filters * 2, num_filters * 4)
        self.down3 = Down(num_filters * 4, num_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(num_filters * 8, num_filters * 16 // factor)

        # Decoding
        self.up1 = Up(num_filters * 16, num_filters * 8 // factor, bilinear)
        self.up2 = Up(num_filters * 8, num_filters * 4 // factor, bilinear)
        self.up3 = Up(num_filters * 4, num_filters * 2 // factor, bilinear)
        self.up4 = Up(num_filters * 2, num_filters, bilinear)
        self.outconv = nn.Conv3d(num_filters, num_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        # Encoding
        xe1 = self.inconv(x)
        xe2 = self.down1(xe1)
        xe3 = self.down2(xe2)
        xe4 = self.down3(xe3)
        xe5 = self.down4(xe4)

        # Decoding
        xd4 = self.up1(xe5, xe4)
        xd3 = self.up2(xd4, xe3)
        xd2 = self.up3(xd3, xe2)
        xd1 = self.up4(xd2, xe1)
        out = self.outconv(xd1)

        outputs = {'x_out': out}

        return outputs


class PatchDiscriminator(nn.Module):
    """
    Patchwise discriminator architecture for use in a GAN setup. 
    Outputs a probability map, where each element classifies a 16x16x16 patch in the original image.
    For inputs 96x96x64, the output is 6x6x4.
    Also outputs the image representation from the third convolutional layer to be used in th ereconstruction loss, as in Larsen et al. 2016

    num_channels = number of input channels. In our case, this is just 1 due to grayscale images
    num_filters = number of filters in the first convolutional layer, this doubles with every convolutional layer
    """
    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(num_channels, num_filters, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 4),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 8),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_filters * 8, 1, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        intermediate = self.conv1(x)
        out = self.conv2(intermediate)
        return out, intermediate