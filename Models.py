import torch
import torch.nn as nn
from model_parts import *


class VAE3D(nn.Module):
    """
    Full VAE architecture with encoder, latent sampling, and decoder
    """
    def __init__(self, input_shape, latent_dim, num_filters, encoder_config, decoder_config):
        super().__init__()

        self.encoder = Encoder(input_shape, latent_dim, num_filters, encoder_config)
        self.decoder = Decoder(input_shape, latent_dim, num_filters, decoder_config)

    def forward(self, x):
        # First pass through encoder
        mu, logvar = self.encoder(x)

        # Sample from latent distribution using reparametrization
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Pass through decoder
        x_recon = self.decoder(z)

        return x_recon, mu, logvar


class UVAE3D(nn.Module):
    """
    U-shaped VAE architecture with skip connections from encoder layers to decoder
    """
    def __init__(self, input_shape, latent_dim, num_filters):
        super().__init__()

        self.encoder = UEncoder(input_shape, latent_dim, num_filters)
        self.decoder = UDecoder(input_shape, latent_dim, num_filters)

    def forward(self, x):
        # First pass through encoder
        layer1, layer2, mu, logvar = self.encoder(x)

        # Sample from latent distribution using reparametrization
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Pass through decoder
        x_recon = self.decoder(z, layer1, layer2)

        return x_recon, mu, logvar

class VQVAE3D(nn.Module):
    """
    Replace the latent layer with vector quantization as described in Oord et al. 2017
    Also include skip connections
    """
    def __init__(self, num_channels, num_filters, embedding_dim = 32, num_embeddings = 512, skip_connections = True):
        super().__init__()
        self.skip = skip_connections
        self.encoder = VQEncoder(num_channels, num_filters, embedding_dim, skip_connections = skip_connections)
        self.quantization = VectorQuantizerEMA(num_embeddings = num_embeddings, embedding_dim = embedding_dim)
        if skip_connections:
            self.decoder = VQDecoder_skip(num_channels, num_filters, embedding_dim)
        else:
            self.decoder = VQDecoder(num_channels, num_filters, embedding_dim)

    def forward(self, x):
        if self.skip:
            x1, x2, x3, ze = self.encoder(x)
            loss, zq, perplexity, _ = self.quantization(ze)
            x_recon = self.decoder(zq, x1, x2, x3)
        else:
            ze = self.encoder(x)
            loss, zq, perplexity, _ = self.quantization(ze)
            x_recon = self.decoder(zq)

        return loss, x_recon, perplexity

class UVAEtop(nn.Module):
    """
    U-shaped VAE architecture that's only trained to recreate just the missing part of an image
    """
    pass


class CropDiscriminator(nn.Module):
    """
    Fully convolutional network that uses 1x1 convolutions and convolutional layers
    to classify only the cropped part of a generated image as good brain reconstruction or bad

    Code is largely based off the DCGAN PyTorch tutorial
    """
    def __init__(self, input_shape, num_filters):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace = True)
        )

        flattened_size = input_shape[1] // 16 * input_shape[2] // 16 * input_shape[3] // 16 * num_filters * 8

        self.fc = nn.Sequential(
                nn.Linear(flattened_size, 1),
                nn.Sigmoid()
        )

        
        
    def forward(self, x):
        x = self.convnet(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        out = self.fc(x)
        return out

class UNet(nn.Module):
    """
    U-shaped VQVAE architecture with skip connections from encoder layers to decoder, 
    similar to original U-net by Ronneberger et al. 2015, but not the exact same.
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

        return out


class Discriminator(nn.Module):
    """
    Patchwise discriminator architecture for use in a GAN setup
    """
    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv3d(num_channels, num_filters, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 4),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm3d(num_filters * 8),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_filters * 8, 1, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        return self.convnet(x)


class AVIDDiscriminator(nn.Module):
    """
    Discriminator network as used in Sabokrou et al. 2019
    Borrowed from https://github.com/masoudpz/AVID-Adversarial-Visual-Irregularity-Detection/blob/master/model.py
    """
    pass