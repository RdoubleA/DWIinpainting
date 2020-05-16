import torch
import torch.nn as nn

"""
This function takes the arguments from the Encoder and Decoder init functions
and sets up a simple 2 layer VAE architecture
"""
def encoder_simple_linear_noactivation(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
            )
    flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4 * num_filters * 2)
        
    layers['fc1'] = nn.Sequential(
        nn.Linear(flattened_size, latent_dim * 2),
        nn.ReLU()
    )
    layers['fc2_mu'] = nn.Linear(latent_dim * 2, latent_dim)
    layers['fc2_logvar'] = nn.Linear(latent_dim * 2, latent_dim)

    return layers

"""
Same as above except linear layers before latent space use tanh activation
"""
def encoder_simple_linear_tanh(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
            )
    flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4 * num_filters * 2)
        
    layers['fc1'] = nn.Sequential(
        nn.Linear(flattened_size, latent_dim * 2),
        nn.Tanh()
    )
    layers['fc2_mu'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.Tanh()
    )
    layers['fc2_logvar'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.Tanh()
    )

    return layers


"""
Same as above except linear layers before latent space use LeakyReLU activation
"""
def encoder_simple_linear_leakyrelu(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
            )
    flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4 * num_filters * 2)
        
    layers['fc1'] = nn.Sequential(
        nn.Linear(flattened_size, latent_dim * 2),
        nn.LeakyReLU()
    )
    layers['fc2_mu'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.LeakyReLU()
    )
    layers['fc2_logvar'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.LeakyReLU()
    )

    return layers

"""
Same as above except linear layers before latent space use LeakyReLU activation
"""
def encoder_simple_linear_relu(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
            )
    flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4 * num_filters * 2)
        
    layers['fc1'] = nn.Sequential(
        nn.Linear(flattened_size, latent_dim * 2),
        nn.ReLU()
    )
    layers['fc2_mu'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.ReLU()
    )
    layers['fc2_logvar'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.ReLU()
    )

    return layers

"""
This function takes the arguments from the Encoder and Decoder init functions
and sets up a simple 2 layer VAE architecture
"""
def decoder_simple_linear_relu(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters, input_shape[0], kernel_size = 2, stride = 2, padding = 0)
            )
    
    flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4 * num_filters * 2)
    
    layers['reshape'] = (input_shape[1] // 4, input_shape[2] // 4, input_shape[3] // 4)
    
    layers['fc1'] = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU()
        )
    layers['fc2'] = nn.Sequential(
            nn.Linear(latent_dim * 2, flattened_size),
            nn.ReLU()
        )

    return layers

"""
Same as above except linear layers use leaky relu
"""
def decoder_simple_linear_leakyrelu(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters, input_shape[0], kernel_size = 2, stride = 2, padding = 0)
            )
    
    flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4 * num_filters * 2)
    
    layers['reshape'] = (input_shape[1] // 4, input_shape[2] // 4, input_shape[3] // 4)
    
    layers['fc1'] = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU()
        )
    layers['fc2'] = nn.Sequential(
            nn.Linear(latent_dim * 2, flattened_size),
            nn.LeakyReLU()
        )

    return layers

"""
Same as above except linear layers use tanh
"""
def decoder_simple_linear_tanh(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters, input_shape[0], kernel_size = 2, stride = 2, padding = 0)
            )
    
    flattened_size = int(input_shape[1] / 4 * input_shape[2] / 4 * input_shape[3] / 4 * num_filters * 2)
    
    layers['reshape'] = (input_shape[1] // 4, input_shape[2] // 4, input_shape[3] // 4)
    
    layers['fc1'] = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh()
        )
    layers['fc2'] = nn.Sequential(
            nn.Linear(latent_dim * 2, flattened_size),
            nn.Tanh()
        )

    return layers

"""
This function takes the arguments from the Encoder and Decoder init functions
and sets up a 4 layer VAE architecture
"""
def encoder_deeper(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU()
        )

    flattened_size = int(input_shape[1] / 16 * input_shape[2] / 16 * input_shape[3] / 16 * num_filters * 8)
        
    layers['fc1'] = nn.Sequential(
        nn.Linear(flattened_size, latent_dim * 2),
        nn.ReLU()
    )
    layers['fc2_mu'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.ReLU()
    )
    layers['fc2_logvar'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.ReLU()
    )

    return layers

"""
This function takes the arguments from the Encoder and Decoder init functions
and sets up a 4 layer VAE architecture
"""
def decoder_deeper(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.ConvTranspose3d(num_filters * 8, num_filters * 4, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters * 4, num_filters * 2, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters * 2, num_filters, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose3d(num_filters, input_shape[0], kernel_size = 2, stride = 2, padding = 0)
            )
    
    flattened_size = int(input_shape[1] / 16 * input_shape[2] / 16 * input_shape[3] / 16 * num_filters * 8)
    
    layers['reshape'] = (input_shape[1] // 16, input_shape[2] // 16, input_shape[3] // 16)
    
    layers['fc1'] = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU()
        )
    layers['fc2'] = nn.Sequential(
            nn.Linear(latent_dim * 2, flattened_size),
            nn.ReLU()
        )

    return layers


"""
This function takes the arguments from the Encoder and Decoder init functions
and sets up a 4 layer VAE architecture with maxpooling
"""
def encoder_deeper_maxpool(input_shape, latent_dim, num_filters):
    layers = {}
    layers['convnet'] = nn.Sequential(
            nn.Conv3d(input_shape[0], num_filters, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size = 2, stride = 2),
            nn.Conv3d(num_filters, num_filters * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size = 2, stride = 2),
            nn.Conv3d(num_filters * 2, num_filters * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size = 2, stride = 2),
            nn.Conv3d(num_filters * 4, num_filters * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size = 2, stride = 2),
        )

    flattened_size = int(input_shape[1] / 16 * input_shape[2] / 16 * input_shape[3] / 16 * num_filters * 8)
        
    layers['fc1'] = nn.Sequential(
        nn.Linear(flattened_size, latent_dim * 2),
        nn.ReLU()
    )
    layers['fc2_mu'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.ReLU()
    )
    layers['fc2_logvar'] = nn.Sequential(
        nn.Linear(latent_dim * 2, latent_dim),
        nn.ReLU()
    )

    return layers
