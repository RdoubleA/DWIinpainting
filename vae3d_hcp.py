import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat
from nilearn.image import index_img
import pickle

from keras.models import Model
from keras.layers import Dense, Flatten, UpSampling3D, Input, Lambda, Reshape, ZeroPadding3D
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model, multi_gpu_model
from keras import backend as K

# ===============================================================
# Specify these, recall that there are about 50000 total samples
# ===============================================================
num_train = 40000
batch_size = 128
crop = True
num_filters = 4 # starting number of filters
latent_dim = 16 # dimensionality of latent space
conv_kernel_size = 3
pool_kernel_size = 2
conv_strides = 2
# ===============================================================
# end
# ===============================================================

"""
Generator function that loads a batch of nifti images as numpy arrays
"""
def image_generator_augmented(file_list, batch_size = 64, crop = False):
    while True:
        # Randomly choose files for the batch
        batch_paths = np.random.choice(a = file_list, size = batch_size)
        batch_input = []
        batch_output = []
        
        # Read in each file, preprocess
        for file_path in batch_paths:
            dwi = np.load(file_path).astype(np.float32)
            if crop:
                dwi_pad = dwi[:,:,-48:, np.newaxis]
            else:
                # Zero pad to make z-dimension divisible by 2, will make convolutional layers work better
                dwi_pad = np.pad(dwi,pad_width=((0,0),(0,0),(0,1)))[:,:,:,np.newaxis]
            batch_input.append(dwi_pad)
            batch_output.append(dwi_pad)
        
        # Return a tuple of (input, output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield (batch_x, batch_y)

def sampling(args):
    mu, sigma = args
    batch_size = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape = (batch_size, dim))
    return mu + K.exp(sigma * 0.5) * eps

# Loss function
def kl_reconstruction_loss(true, pred):
    recon_loss = K.mean(binary_crossentropy(true,pred), axis=[1,2,3])
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis = -1)
    kl_loss *= -0.5
    return K.mean(recon_loss + kl_loss)



dwi_files_npy = np.loadtxt('X_files.txt', dtype=str)
num_val = dwi_files_npy.shape[0] - num_train
input_shape = (144, 168, 48, 1)
train_data_gen = image_generator_augmented(dwi_files_npy[:num_train], batch_size = batch_size, crop = crop)
val_data_gen = image_generator_augmented(dwi_files_npy[num_train:], batch_size = batch_size, crop = crop)
val_steps = np.ceil(num_val / batch_size)
train_steps = np.ceil(num_train / batch_size)

# Delete existing model and clear session
K.clear_session()

# Encoder
inputs = Input(shape=input_shape)
x = inputs
for i in range(2):
    filters = num_filters * (2 ** i)
    x = Conv3D(filters=filters,
               kernel_size=conv_kernel_size,
               activation='relu',
               strides=conv_strides,
               padding='same')(x)
    #x = MaxPooling3D(pool_size = pool_kernel_size)(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(latent_dim*2, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae3d_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3] * shape[4], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3], shape[4]))(x)

for i in range(2):
    filters = num_filters * (2 ** (2-i))
    x = Conv3DTranspose(filters=filters,
                        kernel_size=conv_kernel_size,
                        activation='relu',
                        strides=conv_strides,
                        padding='same')(x)
    #x = UpSampling3D(size = pool_kernel_size)(x)
    

outputs = Conv3DTranspose(filters=1,
                          kernel_size=conv_kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae3d_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')
vae.summary()
vae.compile(optimizer = 'adam', loss = kl_reconstruction_loss)
history = vae.fit_generator(train_data_gen,
              steps_per_epoch = train_steps,
              epochs = 1,
              validation_data = val_data_gen,
              validation_steps = val_steps,
              verbose = 1)

vae.save_weights('vae3d_hcp.h5')
with open('train_history.p', 'wb') as f:
	pickle.dump(history, f)
