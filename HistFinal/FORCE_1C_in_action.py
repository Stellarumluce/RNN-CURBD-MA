# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:08:17 2024

@author: esnow
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from skimage.transform import resize
import math

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
        resized_data[:, :, t] = resized_frame
    return resized_data

# Function to normalize data
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)

# Function to initialize variables
def initialize_variables(N, g, p, alpha):
    scale = 1.0 / np.sqrt(p * N)
    sparse_matrix = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn)
    J_d = sparse_matrix.toarray() * g * scale
    u_in = np.random.uniform(-1, 1, size=(N, N))
    u = np.random.uniform(-1, 1, size=(N, N))
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
retinal_data = waves[1]  # Using the second wave file (index starts from 0)

# Determine Frame Dimensions
frame_height, frame_width, T = retinal_data.shape
print(f"Frame dimensions: {frame_height}x{frame_width}, Number of frames: {T}")

# Load the excitatory data
exc_data = np.load('D:/MA/activity_code/activity_code/data/rates_exc_Htriggered_Hampdirect_trial0.npy')

# Downsample the excitatory data to match the retinal data's temporal resolution
original_dt = 1  # Original temporal resolution of exc_data is 1 ms
new_dt = 10    # New temporal resolution is 10 ms
exc_data_downsampled = downsample_data(exc_data, original_dt, new_dt)

# Resizing the excitatory data frames from 40x40 to 10x10
new_size = (40, 40)
T_resized = retinal_data.shape[2]
resized_exc_data = np.zeros((new_size[0] * new_size[1], T_resized))

for t in range(T_resized):
    frame = exc_data_downsampled[:, t].reshape(40, 40)  # Assuming original size of exc_data frames is 40x40
    resized_frame = resize(frame, new_size, anti_aliasing=True, mode='reflect')
    resized_exc_data[:, t] = resized_frame.reshape(new_size[0] * new_size[1])

print(f"Resized excitatory data dimensions: {resized_exc_data.shape}")

# Resizing the retinal data frames from 64x64 to 10x10
resized_retinal_data = resize_data(retinal_data, new_size)
print(f"Resized retinal data dimensions: {resized_retinal_data.shape}")

# Normalize the data
normalized_exc_data = normalize_data(resized_exc_data)
normalized_retinal_data = normalize_data(resized_retinal_data)
a = normalized_exc_data
# Vectorize retinal data
vectorized_retinal_data = normalized_retinal_data.reshape(new_size[0] * new_size[1], T_resized)
N, T = vectorized_retinal_data.shape
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
dt = 1
tau = 1
t_vec = np.linspace(0, T, int(T/dt))

# Initialize variables
W = np.zeros((N, N))
# Assuming p, N, and g are defined
p = 1 # example value for p

# Calculate scale
scale = 1.0 / np.sqrt(p * N)

# Generate a sparse random matrix with normally distributed values
# The random function generates values in [0, 1), so we need to use a normal distribution
sparse_matrix = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn)

# Convert to dense matrix and scale
J = sparse_matrix.toarray() * g * scale
P = np.eye(N) / alpha  # Initialize learning update matrix P
x = np.zeros((N, len(t_vec)))
x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn()

x[:, 0] = x_0
r = np.tanh(x_0)
z = z_0
ps_vec = []
u = np.random.uniform(-1, 1, size=(N))
# Training Loop only time loop
for t in range(1, len(t_vec)):
    # Update neuron state
    x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J, r)) +  u * vectorized_retinal_data[:, t-1]
    
    # Update firing rate
    r = np.tanh(x[:, t])  # Ensure r is (N,)
    
    # Update network output
    z = np.dot(W.T, r)  # Ensure z is a scalar
    
    # Compute error
    e_minus = z - a[:,t-1]
    
    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = 1.0 / (1.0 + rPr)
    W -= c*np.outer(k, e_minus) #key training process #sensitive to errorc * np.outer(k, e)
    P = P - c * np.outer(k, k)
    J = J - np.outer(e_minus, k)*c
 

# Testing Phase
simtime_len = len(t_vec)
zpt = np.zeros((N,simtime_len))
x_test = x[:, 0]  # Start from the last state of the training phase
r_test = np.tanh(x_test)
z = z_0
for ti in range(simtime_len):
    x_test = (1.0 - dt) * x_test + np.dot(J, r_test * dt) 
    r_test = np.tanh(x_test)
    z = np.dot(W.T, r_test)  # Use the final trained weights
    zpt[:, ti] = z


error_avg = np.mean(np.abs(zpt - a))
print(f'Testing MAE: {error_avg:.3f}')


for iter in range(simtime_len):
    # Create a new figure with a custom size for better scaling
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Adjust figsize for larger plots

    # Plot the output frame
    img_1 = axes[0].imshow(zpt[:, iter].reshape(new_size), cmap='viridis')
    axes[0].set_title(f'Output Frame {iter} FORCE 1C')
    cbar_1 = fig.colorbar(img_1, ax=axes[0])  # Add colorbar to the first subplot

    # Plot the target frame
    img_2 = axes[1].imshow(a[:, iter].reshape(new_size), cmap='viridis')
    axes[1].set_title(f'Target Frame {iter}')
    cbar_2 = fig.colorbar(img_2, ax=axes[1])  # Add colorbar to the second subplot

    # Adjust layout to prevent overlapping of colorbars and plots
    plt.tight_layout(pad=2)  # You can adjust pad as needed
    
    # Show the plot
    plt.show()
