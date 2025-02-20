# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:40:09 2024

@author: esnow
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from skimage.transform import resize

# Function to load retinal wave data
def load_retinal_wave_data(wave_directory):
    wave_files = [os.path.join(wave_directory, f) for f in os.listdir(wave_directory) if f.endswith('.npy')]
    waves = [np.load(wave_file) for wave_file in wave_files]
    return waves

# Function to resize data
def resize_data(data, new_size):
    T = data.shape[2]
    resized_data = np.zeros((new_size[0], new_size[1], T))
    for t in range(T):
        resized_frame = resize(data[:, :, t], new_size, anti_aliasing=True, mode='reflect')
        threshold = 0.01
        resized_data[:, :, t] = (resized_frame > threshold).astype(int)
    return resized_data

# Function to normalize data
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Function to initialize variables
def initialize_variables(N, g, p, alpha):
    scale = 1.0 / np.sqrt(p * N)
    sparse_matrix = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn)
    J_d = sparse_matrix.toarray() * g * scale
    u_in = np.random.uniform(-1, 1, size=(N))
    u = np.random.uniform(-1, 1, size=(N))
    P = np.eye(N) / alpha
    x_0 = 0.5 * np.random.randn(N)
    return J_d, u_in, u, P, x_0

# Function to downsample data
def downsample_data(data, original_dt, new_dt):
    factor = int(new_dt / original_dt)
    return data[:, ::factor]

# Load Retinal Wave Data
wave_directory = 'data/generated_waves/Trial0_Directional/numpy_waves/'
waves = load_retinal_wave_data(wave_directory)
retinal_data = waves[1]  # Using the fifth wave file

# Determine Frame Dimensions
frame_height, frame_width, T = retinal_data.shape
print(f"Frame dimensions: {frame_height}x{frame_width}, Number of frames: {T}")

# Load the excitatory data
exc_data = np.load('D:/MA/activity_code/activity_code/data/rates_exc_Htriggered_Hampdirect_trial0.npy')
inh_data = np.load('D:/MA/activity_code/activity_code/data/rates_inh_Htriggered_Hampdirect_trial0.npy')

# Downsample the excitatory data to match the retinal data's temporal resolution
original_dt = 1  # Original temporal resolution of exc_data is 1 ms
new_dt = 10  # New temporal resolution should match retinal data, which is 0.5 seconds (500 ms)
exc_data_downsampled = downsample_data(exc_data, original_dt, new_dt)

# Resizing the excitatory data frames from 40x40 to 20x20
new_size = (10, 10)
T_resized = retinal_data.shape[2]
resized_exc_data = np.zeros((new_size[0] * new_size[1], T_resized))

for t in range(T_resized):
    frame = exc_data_downsampled[:, t].reshape(40, 40)  # Assuming original size of exc_data frames is 40x40
    resized_frame = resize(frame, new_size, anti_aliasing=True, mode='reflect')
    resized_exc_data[:, t] = resized_frame.reshape(new_size[0] * new_size[1])

print(f"Resized excitatory data dimensions: {resized_exc_data.shape}")

# Resizing the retinal data frames from 64x64 to 20x20
resized_retinal_data = resize_data(retinal_data, new_size)
print(f"Resized retinal data dimensions: {resized_retinal_data.shape}")

# Normalize the data
normalized_exc_data = normalize_data(resized_exc_data)
normalized_retinal_data = normalize_data(resized_retinal_data)

# Vectorize retinal data
vectorized_retinal_data = normalized_retinal_data.reshape(new_size[0] * new_size[1], T_resized)
# Assuming vectorized_retinal_data is already defined and has shape (N, T_resized)
N, T_resized = vectorized_retinal_data.shape

# Generate Gaussian noise with mean 0 and standard deviation 1
noise = np.random.normal(0, 1, (N, T_resized))

# # Optionally, normalize the noise to have a similar scale as the retinal data
# noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

# Parameters
N = new_size[0] * new_size[1]
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
dt = 0.5
tau = 1
t_vec = np.linspace(0, T_resized, T_resized)  # For temporal convenience for now
W = np.zeros((N))  # for each element of 400 we have a designated w vector

# Initialize variables
p = 1  # Sparsity
J_d, u_in, u, P, x_0 = initialize_variables(N, g, p, alpha)

x = np.zeros((N, len(t_vec)))
x_d = np.zeros((N, len(t_vec)))
x[:, 0] = x_0
x_d[:, 0] = x_0
r = np.tanh(x_0)
r_d = np.tanh(x_0)
J_N = np.zeros((N, N))
J_pre_N = np.zeros((N, N))
W_pre = np.zeros((N))

M = np.eye(N)
# Training Loop
for iter in range(1):
    J_N = J_pre_N
    W = W_pre
    for neuron_num in range(N):
        x = np.zeros((N, len(t_vec)))
        x_d = np.zeros((N, len(t_vec)))
        x[:, 0] = x_0
        x_d[:, 0] = x_0
        P = np.eye(N) / alpha  # Initialize learning update matrix P
        J_d = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn).toarray() * g / np.sqrt(p * N)
        r = np.tanh(x_0)
        r_d = np.tanh(x_0)

        for t in range(1, len(t_vec)):
            x_d[:, t] = (1 - dt / tau) * x_d[:, t-1] + (dt / tau) * (np.dot(J_d, r_d) + u * resized_exc_data[neuron_num, t-1] + u*vectorized_retinal_data[neuron_num, t-1])
            x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J_N, r) + u * vectorized_retinal_data[neuron_num, t-1])
            r_d = np.tanh(x_d[:, t])
            r = np.tanh(x[:, t])
            
            e = np.dot(J_N, r) - np.dot(J_d, r_d) - resized_exc_data[neuron_num, t-1]
            k = np.dot(P, r)
            rPr = np.dot(r.T, k)
            c = 1.0 / (1.0 + rPr)
            
            J_N = J_N - np.outer(e, k) * c
            z = np.dot(W.T, r)
            e_w = z - resized_exc_data[neuron_num, t-1]
            W = W - e_w * k * c  # Key training process
            P = P - c * np.outer(k, k)
            J_pre_N = J_N
            W_pre = W

# Testing Phase
simtime_len = len(t_vec)
zpt = np.zeros(simtime_len)
x_test = x[:, -1]  # Start from the last state of the training phase
r_test = np.tanh(x_test)
final_zpt = np.zeros((N, simtime_len))
for neuron_num in range(N):
    zpt = np.zeros(simtime_len)
    x_test = x[:, -1]  # Start from the last state of the training phase
    r_test = np.tanh(x_test)
    for ti in range(simtime_len):
        x_test = (1 - dt / tau) * x_test + (dt / tau) * (np.dot(J_pre_N, r_test) + u *vectorized_retinal_data[neuron_num, ti-1])
        r_test = np.tanh(x_test)
        z = np.dot(W_pre.T, r_test)  # Use the final trained weights
        zpt[ti] = z
    final_zpt[neuron_num, :] = zpt.T
# Resync by observation
final_zpt_resyn = final_zpt[:,1:]
a_resyn = resized_exc_data[:, :T-1]
error_avg = np.mean(np.abs(final_zpt_resyn - a_resyn))


print(f'Testing MAE: {error_avg:.3f}')
# Save plots for each iteration
for iter in range(T-1):
    # Plot the results
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    img1 = plt.imshow(final_zpt_resyn[:, iter].reshape(10, 10), aspect='auto')
    plt.title('Output Frame')
    plt.colorbar(img1)

    plt.subplot(1, 2, 2)
    img2 = plt.imshow(a_resyn[:, iter].reshape(10, 10), aspect='auto')
    plt.title('Target Frame')
    plt.colorbar(img2)
    plt.show()
    plt.tight_layout()