# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 00:07:24 2024

@author: esnow
"""

import numpy as np
import os
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from scipy.sparse import random as sparse_random
from skimage.transform import resize

# Load Retinal Wave Data
wave_directory =os.path.expanduser('~/NAS/projects/2024VisDev/wave_data/generated_waves/Trial0_Directional/numpy_waves/')
wave_files = [os.path.join(wave_directory, f) for f in os.listdir(wave_directory) if f.endswith('.npy')]
waves = [np.load(wave_file) for wave_file in wave_files]
retinal_data = waves[0]  # Assuming using the first wave file

# Determine Frame Dimensions
frame_height, frame_width, T = retinal_data.shape
print(f"Frame dimensions: {frame_height}x{frame_width}, Number of frames: {T}")

# Load excitatory and inhibitory data
exc_data = np.load(os.path.expanduser('~/NAS/projects/2024VisDev/shreya_data/rates_exc_Htriggered_Hampdirect_trial0.npy'))
# inh_data = np.load('D:/MA/activity_code/activity_code/data/rates_inh_Htriggered_Hampdirect_trial0.npy')
N = exc_data.shape[0]
print(f"Excitatory data dimensions: {exc_data.shape}")

# Resizing the retinal data frames from 64x64 to 40x40
new_size = (40, 40)
resized_retinal_data = np.zeros((new_size[0], new_size[1], T))

for t in range(T):
    resized_frame = resize(retinal_data[:, :, t], new_size, anti_aliasing=True, mode='reflect')
    threshold = 0.001
    resized_retinal_data[:, :, t] = (resized_frame > threshold).astype(int)
# Verify new dimensions
print(f"Resized retinal data dimensions: {resized_retinal_data.shape}")
vectorized_retinal_data = resized_retinal_data.reshape(new_size[0] * new_size[1], T)

a = exc_data[:, :T]  # Adjust size to match the RNN training duration

## Set the output directory
output_dir = os.path.expanduser('~/NAS/projects/2024VisDev/FF_plots_1600_Synthetic0')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Parameters
N = exc_data.shape[0]
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
dt = 0.5
tau = 1
t_vec = np.linspace(0, T, T) #For temporal convenience for now
W = np.zeros((N, N)) # for each element of 1600 we have a destinated w vector
# Initialize variables
p = 1  # Sparsity
scale = 1.0 / np.sqrt(p * N)

# Generate a sparse random matrix with normally distributed values
sparse_matrix = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn)

# Convert to dense matrix and scale
J_d = sparse_matrix.toarray() * g * scale
u = np.random.uniform(-1, 1, size=(N))
P = np.eye(N) / alpha  # Initialize learning update matrix P
x = np.zeros((N, len(t_vec)))
x_d = np.zeros((N, len(t_vec)))
x_0 = 0.5 * np.random.randn(N)
x[:, 0] = x_0
x_d[:, 0] = x_0
r = np.tanh(x_0)
r_d = np.tanh(x_0)
J_N = np.zeros((N,N,N))
J_pre_N = np.zeros((N,N,N))
W_pre = np.zeros((N,N))    

M = np.eye(N)
# Training Loop
for iter in range(1):
    J_N = J_pre_N.copy()
    W = W_pre.copy()
    for neuron_num in range(N):
        # neuron_num = 10
        x = np.zeros((N, len(t_vec)))
        x_d = np.zeros((N, len(t_vec)))
        x[:, 0] = x_0
        x_d[:, 0] = x_0
        P = np.eye(N) / alpha  # Initialize learning update matrix P
        J_d = sparse_matrix.toarray() * g * scale
        r = np.tanh(x_0)
        r_d = np.tanh(x_0)

        for t in range(1, len(t_vec)):
            # Update neuron state
            x_d[:, t] = (1 - dt / tau) * x_d[:, t-1] + (dt / tau) * (np.dot(J_d, r_d) + u * a[neuron_num,t-1] + u * vectorized_retinal_data[neuron_num,t-1])
            x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J_N[:,:,neuron_num], r) + u *  vectorized_retinal_data[neuron_num,t-1])
            # Update firing rate
            r_d = np.tanh(x_d[:, t])  # Ensure r_d is (N,)
            r = np.tanh(x[:, t])  # Ensure r is (N,)
            
            # Compute error for recurrent weights
            e = np.dot(J_N[:,:,neuron_num], r) - np.dot(J_d, r_d) - u * a[neuron_num,t-1]
        
            k = np.dot(P, r)
            rPr = np.dot(r.T, k)
            c = 1.0 / (1.0 + rPr)
        
            
            J_N[:,:,neuron_num] = J_N[:,:,neuron_num] - np.outer(e, k)*c
            # Compute network output
            z = np.dot(W[:, neuron_num].T, r)
            
            # Update readout weights
            e_w = z - a[neuron_num,t-1]  # Error in the output
            # RLS update
            
            W[:, neuron_num] = W[:, neuron_num] - e_w * k * c #key training process
            P = P - c * np.outer(k, k)
            J_pre_N[:,:,neuron_num] = J_N[:,:,neuron_num]
            W_pre[:, neuron_num] = W[:, neuron_num]
# Testing Phase
simtime_len = len(t_vec)
zpt = np.zeros(simtime_len)
x_test = x[:, -1]  # Start from the last state of the training phase
r_test = np.tanh(x_test)
final_zpt = np.zeros((N,simtime_len))
for neuron_num in range(N):
    zpt = np.zeros(simtime_len)
    x_test = x[:, -1]  # Start from the last state of the training phase
    r_test = np.tanh(x_test)
    for ti in range(simtime_len):
        x_test = (1 - dt / tau) * x_test + (dt / tau) * (np.dot(J_pre_N[:,:,neuron_num], r_test) + u * a[neuron_num,ti-1])
        r_test = np.tanh(x_test)
        z = np.dot(W_pre[:, neuron_num].T, r_test)  # Use the final trained weights
        zpt[ti] = z
    final_zpt[neuron_num, :] = zpt.T
error_avg = np.mean(np.abs(final_zpt - a))
# Ensure that final_zpt and a are numpy arrays
final_zpt = np.array(final_zpt)
a = np.array(a)

print(f'Testing MAE: {error_avg:.3f}')
# Save plots for each iteration
for iter in range(T):
    # Plot the results
    plt.figure()
    plt.subplot(1, 2, 1)
    img1 = plt.imshow(final_zpt[:, iter].reshape(40, 40), aspect='auto')
    plt.title(f'Output Frame {iter}')
    plt.colorbar(img1)

    plt.subplot(1, 2, 2)
    img2 = plt.imshow(a[:, iter].reshape(40, 40), aspect='auto')
    plt.title(f'Target Frame {iter}')
    plt.colorbar(img2)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'plot_{iter}.png'))
    plt.close()
