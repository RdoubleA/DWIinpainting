import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Encoder(nn.Module):
    """
    Encoder architecture of a VAE with a specified number of layers, used in VAE3D
    """
    def __init__(self, input_shape, latent_dim, num_filters, config_function):
        super().__init__()
        layers = config_function(input_shape, latent_dim, num_filters)
        
        self.convnet = layers['convnet']
        self.fc1 = layers['fc1']
        self.fc2_mu = layers['fc2_mu']
        self.fc2_logvar = layers['fc2_logvar']

    def forward(self, input):

        x = self.convnet(input)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = self.fc1(x)
        mu  = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)

        return mu, logvar

class Decoder(nn.Module):
    """
    Decoder architecture of a VAE with a specified number of layers, used in VAE3D
    """
    def __init__(self, input_shape, latent_dim, num_filters, config_function):
        super().__init__()
        layers = config_function(input_shape, latent_dim, num_filters)
        self.reshape = layers['reshape']
        self.fc1 = layers['fc1']
        self.fc2 = layers['fc2']
        self.convnet = layers['convnet']    

    def forward(self, input):

        h = self.fc1(input)
        x = self.fc2(h)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.reshape[0], self.reshape[1], self.reshape[2])
        x_recon = self.convnet(x)

        return x_recon

class UEncoder(nn.Module):
    """
    Encoder architecture that returns intermediary outputs to be used as skip connections
    """
    def __init__(self, input_shape, latent_dim, num_filters):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
        )
 
        self.conv3 = nn.Sequential(
            nn.Conv3d(num_filters * 2, 1, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )   

        flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4)

        self.fc_mu = nn.Sequential(
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU()
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU()
        )


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        batch_size = x3.shape[0]
        x3_flattened = x3.view(batch_size, -1)
        mu  = self.fc_mu(x3_flattened)
        logvar = self.fc_logvar(x3_flattened)

        return x1, x2, mu, logvar

class UDecoder(nn.Module):
    """
    Decoder architecture that accepts intermediary outputs from encoder as skip connections
    """
    def __init__(self, input_shape, latent_dim, num_filters):
        super().__init__()

        flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4)

        self.fc = nn.Sequential(
                nn.Linear(latent_dim, flattened_size),
                nn.ReLU()
            )

        self.reshape = (input_shape[1] // 4, input_shape[2] // 4, input_shape[3] // 4)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(1, num_filters * 2, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
            )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 4, num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
            )

        self.conv3 = nn.ConvTranspose3d(num_filters * 2, input_shape[0], kernel_size = 2, stride = 2, padding = 0)
        

    def forward(self, x, encoder_layer1_output, encoder_layer2_output):

        h = self.fc(x)
        batch_size = h.shape[0]
        h = h.view(batch_size, -1, self.reshape[0], self.reshape[1], self.reshape[2])
        x1 = self.conv1(h)
        x2 = torch.cat([encoder_layer2_output, x1], dim = 1)
        x3 = self.conv2(x2)
        x4 = torch.cat([encoder_layer1_output, x3], dim = 1)
        x_recon = self.conv3(x4)

        return x_recon


class VQEncoder(nn.Module):
    """
    Encoder architecture that outputs a volume to a quantization layer. Also outputs skip connections
    """
    def __init__(self, num_channels, num_filters = 8, embedding_dim = 32, skip_connections = False):
        super().__init__()

        self.skip = skip_connections

        self.conv1 = nn.Sequential(
            nn.Conv3d(num_channels, num_filters, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(num_filters * 4, embedding_dim, kernel_size = 4, stride = 2, padding = 1)
        )


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        ze = self.conv4(x3)

        if self.skip:
            return x1, x2, x3, ze
        else:
            return ze


class VQDecoder_skip(nn.Module):
    """
    Decoder architecture that accepts a volume from a quantization layer and skip connections from the encoder
    """
    def __init__(self, num_channels, num_filters = 8, embedding_dim = 32):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
            )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 8, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
            )


        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 4, num_filters, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
            )

        self.conv4 = nn.ConvTranspose3d(num_filters * 2, num_channels, kernel_size = 4, stride = 2, padding = 1)
        

    def forward(self, zq, encoder_layer1_output, encoder_layer2_output, encoder_layer3_output):

        x1 = self.conv1(zq)
        x2 = torch.cat([encoder_layer3_output, x1], dim = 1)
        x3 = self.conv2(x2)
        x4 = torch.cat([encoder_layer2_output, x3], dim = 1)
        x5 = self.conv3(x4)
        x6 = torch.cat([encoder_layer1_output, x5], dim = 1)
        x_recon = self.conv4(x6)

        return x_recon


class VQDecoder(nn.Module):
    """
    Decoder architecture that accepts a volume from a quantization layer 
    """
    def __init__(self, num_channels, num_filters = 8, embedding_dim = 32):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, num_filters * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
            )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 4, num_filters * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
            )


        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
            )

        self.conv4 = nn.ConvTranspose3d(num_filters, num_channels, kernel_size = 4, stride = 2, padding = 1)
        

    def forward(self, zq):

        x1 = self.conv1(zq)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_recon = self.conv4(x3)

        return x_recon


class DoubleConv(nn.Module):
    """
    Runs two convolutional layers, similar to the U-Net implementation, used in UNet
    """
    def __init__(self, filters_in, filters_out):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv3d(filters_in, filters_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(filters_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters_out, filters_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(filters_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input):
        return self.convnet(input)


class Down(nn.Module):
    """
    Max pool then pass through two convolutional layers, used in UNet
    """
    def __init__(self, filters_in, filters_out):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.MaxPool3d(2, stride = 2),
            DoubleConv(filters_in, filters_out)
        )

    def forward(self, x):
        return self.convnet(x)


class Up(nn.Module):
    """
    Transpose convolution to upsample, then double convolution, used in UNet
    """
    def __init__(self, filters_in, filters_out, bilinear = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'trilinear', align_corners = True)
        else:
            self.up = nn.ConvTranspose3d(filters_in, filters_out, kernel_size = 2, stride = 2)
        
        self.conv = DoubleConv(filters_in, filters_out)

    def forward(self, x1, x2):
        # Requires a feature-dimension concatenation of the output from an encoding layer at the same level
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class PixelShuffle3D(nn.Module):
    """
    Last step of Subpixel Deconvolution, as described in Shi et al. 2016. 
    Taken from authors of Tudosiu et al. 2020 at https://github.com/danieltudosiu/nmpevqvae/blob/master/network.py
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor
        self.upscale_factor_cubed = upscale_factor ** 3
        self._shuffle_out = None
        self._shuffle_in = None

    def forward(self, input):
        shuffle_out = input.new()

        batch_size = input.size(0)
        channels = int(input.size(1) / self.upscale_factor_cubed)
        in_depth = input.size(2)
        in_height = input.size(3)
        in_width = input.size(4)

        input_view = input.view(
            batch_size,
            channels,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )

        shuffle_out.resize_(
            input_view.size(0),
            input_view.size(1),
            input_view.size(5),
            input_view.size(2),
            input_view.size(6),
            input_view.size(3),
            input_view.size(7),
            input_view.size(4),
        )

        shuffle_out.copy_(input_view.permute(0, 1, 5, 2, 6, 3, 7, 4))

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        output = shuffle_out.view(
            batch_size, channels, out_depth, out_height, out_width
        )

        return output


class SubPixel3D(nn.Module):
    """
    Performs SubPixel Deconvolution to upsample a 3D image, according to Shi et al. 2016
    """
    def __init__(self, filters_in, filters_out, upsampling_factor):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv3d(filters_in, filters_out * 2 ** 3, kernel_size = 3, stride = 1, padding = 1),
            PixelShuffle3D(upsampling_factor)
            )

    def forward(self, x):
        return self.convnet(x)


class VectorQuantizerEMA(nn.Module):
    """
    Vector quantization layer as implemented in Oord et al. 2017 with exponential moving average updates
    Taken from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

    Tudosiu et al. 2020 uses a commitment cost of 7 for 192x256x192 T1 images from ADNI. For comparison, 
    the original paper use 0.25 for CIFAR10 which is 32x32. I may need something slightly smaller.
    cost since our images are smaller.

    I've also used the K and D values from the deepest quantization layer in Tudosiu et al. as defaults
    """
    def __init__(self, num_embeddings = 256, embedding_dim = 32, commitment_cost = 6, decay = 0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHWD -> BHWDC
        inputs = inputs.permute(0, 2, 3, 4, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWDC -> BCHWD
        return loss, quantized.permute(0, 4, 1, 2, 3).contiguous(), perplexity, encodings

        

