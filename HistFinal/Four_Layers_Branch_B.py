# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:27:58 2024

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
        threshold = 0.01
        resized_data[:, :, t] = (resized_frame > threshold).astype(int)
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
new_dt = 10   # New temporal resolution is 10 ms
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
WB = np.zeros((N, N))
# Assuming p, N, and g are defined
p = 1 # example value for p
u = np.random.uniform(-1, 1, size=(N))
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
r_1 = r
r_2 = r
r_3 = r
r_4 = r
z = z_0
x_1 = x
x_2 = x
x_3 = x
x_4 = x
J_1 = J
J_2 = J
J_3 = J
J_4 = J
P_1 = P
P_2 = P
P_3 = P
P_4 = P

###Init for branch B
r_B1 = r
r_B2 = r
r_B3 = r
r_B4 = r
z_B = z_0
x_B1 = x
x_B2 = x
x_B3 = x
x_B4 = x
J_B1 = J
J_B2 = J
J_B3 = J
J_B4 = J
P_B1 = P
P_B2 = P
P_B3 = P
P_B4 = P
# Initialize feed-forward connection matrix
C1 = np.random.randn(N, N) / np.sqrt(N)
C2 = np.random.randn(N, N) / np.sqrt(N)
C3 = np.random.randn(N, N) / np.sqrt(N)

# Initialize feed-forward connection matrix
CB1 = np.random.randn(N, N) / np.sqrt(N)
CB2 = np.random.randn(N, N) / np.sqrt(N)
CB3 = np.random.randn(N, N) / np.sqrt(N)
# The data is generated from a normal distribution with a small variance
sparsity = 0.1
rvs = lambda x: np.random.randn(x) / np.sqrt(N)   
C_sparse = sparse_random(N, N, density=sparsity, data_rvs=rvs)
C_sparse = C_sparse.toarray()
# Training Loop only time loop
y_input_TB3 = np.dot(C3, r_3)
y_input_FB3 = np.dot(CB3, r_B3)
for t in range(1, len(t_vec)):
    # Update neuron state
    # x_1[:, t] = (1 - dt / tau) * x_1[:, t-1] + (dt / tau) * (np.dot(J_1, r_1)) + u*vectorized_retinal_data[:, t-1]
    # # Update firing rate
    # r_1 = np.tanh(x_1[:, t])  # Ensure r is (N,)
    # y_input_1 = np.dot(C1, r_1)
    
    x_2[:, t] = (1 - dt / tau) * x_2[:, t-1] + (dt / tau) * (np.dot(J_2, r_2)) 
    # Update firing rate
    r_2 = np.tanh(x_2[:, t])  # Ensure r is (N,)
    
    y_input_2 = np.dot(C2, r_2) #sparsity 0.1 or higher has huger impact but scaling with 0.1 would also degrade learning
    x_3[:, t] = (1 - dt / tau) * x_3[:, t-1] + (dt / tau) * (np.dot(J_3, r_3)) + u * y_input_2 
    # Update firing rate
    r_3 = np.tanh(x_3[:, t])  # Ensure r is (N,)
    
    y_input_3 = np.dot(C3, r_3)
    y_input_FB3 = np.dot(CB3, r_B3)
    x_4[:, t] = (1 - dt / tau) * x_4[:, t-1] + (dt / tau) * (np.dot(J_4, r_4)) + u * y_input_3 + u* y_input_FB3
    # Update firing rate
    r_4 = np.tanh(x_4[:, t])  # Ensure r is (N,)
    # Update network output
    z = np.dot(W.T, r_4)  # Ensure z is a scalar
    
    # Compute error
    e_minus = z - a[:,t-1]
    
    # #### comment out to check the simple mode
    k1 = np.dot(P_1, r_1)
    rPr_1 = np.dot(r_1.T, k1)
    c1 = 1.0 / (1.0 + rPr_1)
    # C1 -=  c1 * np.outer(k1, e_minus)
    J_1 = J_1 - c1 * np.outer(e_minus, k1)
    P_1 = P_1 - c1 * np.outer(k1, k1)
    
    k2 = np.dot(P_2, r_2)
    rPr_2 = np.dot(r_2.T, k2)
    c2 = 1.0 / (1.0 + rPr_2)
    # C2 -=  c2 * np.outer(k2, e_minus)
    J_2 = J_2 - np.outer(e_minus, k2)*c2
    P_2 = P_2 - c2 * np.outer(k2, k2)
    
    
    k3 = np.dot(P_3, r_3)
    rPr_3 = np.dot(r_3.T, k3)
    c3 = 1.0 / (1.0 + rPr_3)
    # C3 -=  c3 * np.outer(k3, e_minus)
    J_3 = J_3 - np.outer(e_minus, k3)*c3
    P_3 = P_3 - c3 * np.outer(k3, k3)
    # ####
    k4 = np.dot(P_4, r_4)
    rPr = np.dot(r_4.T, k4)
    c = 1.0 / (1.0 + rPr)
    W -= c * np.outer(k4, e_minus) #key training process #sensitive to errorc * np.outer(k, e)
    J_4 = J_4 - np.outer(e_minus, k4)*c
    P_4 = P_4 - c * np.outer(k4, k4)
    ##########Branch B Training Process############################################
    # Update neuron state
    # x_B1[:, t] = (1 - dt / tau) * x_B1[:, t-1] + (dt / tau) * (np.dot(J_B1, r_B1)) + u*vectorized_retinal_data[:, t-1]
    # # Update firing rate
    # r_B1 = np.tanh(x_B1[:, t])  # Ensure r is (N,)
    # y_input_B1 = np.dot(CB1, r_B1)
    
    x_B2[:, t] = (1 - dt / tau) * x_B2[:, t-1] + (dt / tau) * (np.dot(J_B2, r_B2)) 
    # Update firing rate
    r_B2 = np.tanh(x_B2[:, t])  # Ensure r is (N,)
    
    y_input_B2 = np.dot(CB2*1, r_B2) #sparsity 0.1 or higher has huger impact but scaling with 0.1 would also degrade learning
    x_B3[:, t] = (1 - dt / tau) * x_B3[:, t-1] + (dt / tau) * (np.dot(J_B3, r_B3)) + u * y_input_B2
    # Update firing rate
    r_B3 = np.tanh(x_B3[:, t])  # Ensure r is (N,)
    
    y_input_B3 = np.dot(CB3, r_B3)
    y_input_TB3 = np.dot(C3, r_3)
    x_B4[:, t] = (1 - dt / tau) * x_B4[:, t-1] + (dt / tau) * (np.dot(J_B4, r_B4)) + u * y_input_B3  + u* y_input_TB3
    # Update firing rate
    r_B4 = np.tanh(x_B4[:, t])  # Ensure r is (N,)
    # Update network output
    z_B = np.dot(W.T, r_B4)  # Ensure z is a scalar
    
    # Compute error
    e_B = z_B - a[:,t-1]
    
    # #### comment out to check the simple mode
    kB1 = np.dot(P_B1, r_B1)
    rPr_B1 = np.dot(r_B1.T, kB1)
    cB1 = 1.0 / (1.0 + rPr_B1)
    # C1 -=  c1 * np.outer(k1, e_minus)
    J_B1 = J_B1 - cB1 * np.outer(e_B, kB1)
    P_B1 = P_B1 - cB1 * np.outer(kB1, kB1)
    
    kB2 = np.dot(P_B2, r_B2)
    rPr_B2 = np.dot(r_B2.T, kB2)
    cB2 = 1.0 / (1.0 + rPr_B2)
    # C2 -=  c2 * np.outer(k2, e_minus)
    J_B2 = J_B2 - np.outer(e_B, kB2)*cB2
    P_B2 = P_B2 - cB2 * np.outer(kB2, kB2)
    
    
    kB3 = np.dot(P_B3, r_B3)
    rPr_B3 = np.dot(r_B3.T, kB3)
    cB3 = 1.0 / (1.0 + rPr_B3)
    # C3 -=  c3 * np.outer(k3, e_minus)
    J_B3 = J_B3 - np.outer(e_B, kB3)*cB3
    P_B3 = P_B3 - cB3 * np.outer(kB3, kB3)
    # ####
    kB4 = np.dot(P_B4, r_B4)
    rPrB = np.dot(r_B4.T, kB4)
    cB = 1.0 / (1.0 + rPrB)
    WB -= cB * np.outer(kB4, e_B) #key training process #sensitive to errorc * np.outer(k, e)
    J_B4 = J_B4 - np.outer(e_B, kB4)*cB
    P_B4 = P_B4 - cB * np.outer(kB4, kB4)
    ############################################################################################################
# Testing Phase
simtime_len = len(t_vec)
zpt = np.zeros((N,simtime_len))
x_test_1 = x_1[:, 0]
x_test_3 = x_3[:, 0]
x_test_2 = x_2[:, 0]
x_test_4 = x_4[:, 0]
r_test_1 = np.tanh(x_test_1)
r_test_3 = np.tanh(x_test_3)
r_test_2 = np.tanh(x_test_2)
r_test_4 = np.tanh(x_test_4)


zptB = np.zeros((N,simtime_len))
x_test_B1 = x_B1[:, 0]
x_test_B3 = x_B3[:, 0]
x_test_B2 = x_B2[:, 0]
x_test_B4 = x_B4[:, 0]
r_test_B1 = np.tanh(x_test_B1)
r_test_B3 = np.tanh(x_test_B3)
r_test_B2 = np.tanh(x_test_B2)
r_test_B4 = np.tanh(x_test_B4)

for ti in range(simtime_len):
    # x_test_1 = (1.0 - dt) * x_test_1 + np.dot(J_1, r_test_1 * dt)  + u*vectorized_retinal_data[:, ti-1]
    # r_test_1 = np.tanh(x_test_1)
    # y_test_input_1 = np.dot(C1, r_test_1)
    x_test_2 = (1.0 - dt) * x_test_2 + np.dot(J_2, r_test_2 * dt) 
    r_test_2 = np.tanh(x_test_2)
    y_test_input_2 = np.dot(C_sparse, r_test_2)
    # y_test_input_2 = np.dot(C2 * 0.1, r_test_2)
    x_test_3 = (1.0 - dt) * x_test_3+ np.dot(J_3, r_test_3 * dt) +  u*y_test_input_2
    r_test_3 = np.tanh(x_test_3)
    y_test_input_3 = np.dot(C3, r_test_3)
    y_test_input_FB3 = np.dot(CB3, r_test_B3)
    x_test_4 = (1.0 - dt) * x_test_4+ np.dot(J_4, r_test_4 * dt) +  u*y_test_input_3 + u*y_test_input_FB3
    r_test_4 = np.tanh(x_test_4)
    zpt[:, ti] = np.dot(W.T, r_test_4)  # Use the final trained weights
    
    # x_test_B1 = (1.0 - dt) * x_test_B1 + np.dot(J_B1, r_test_B1 * dt)  + u*vectorized_retinal_data[:, ti-1]
    # r_test_B1 = np.tanh(x_test_B1)
    # y_test_input_B1 = np.dot(CB1, r_test_B1)
    x_test_B2 = (1.0 - dt) * x_test_B2 + np.dot(J_B2, r_test_B2 * dt) 
    r_test_B2 = np.tanh(x_test_B2)
    y_test_input_B2 = np.dot(CB2, r_test_B2)
    x_test_B3 = (1.0 - dt) * x_test_B3+ np.dot(J_B3, r_test_B3 * dt) +  u*y_test_input_B2
    r_test_B3 = np.tanh(x_test_B3)
    y_test_input_B3 = np.dot(CB3, r_test_B3)
    y_test_input_TB3 = np.dot(C3, r_test_3)
    x_test_B4 = (1.0 - dt) * x_test_B4+ np.dot(J_B4, r_test_B4 * dt) +  u*y_test_input_B3 + u*y_test_input_TB3
    r_test_B4 = np.tanh(x_test_B4)
    zptB[:, ti] = np.dot(WB.T, r_test_B4)  # Use the final trained weights
   

error_avg_v = np.mean(np.abs(zpt - a))
error_avg_d = np.mean(np.abs(zptB - a))
print(f'Testing MAE vHVA: {error_avg_v:.3f}')
print(f'Testing MAE dHVA: {error_avg_d:.3f}')

for iter in range(simtime_len):
    # Create a figure and 2x2 layout of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Plot on each subplot
    axes[0, 0].imshow(zpt[:, iter].reshape(new_size))
    axes[0, 0].set_title(f'vHVA Output frame {iter}')
    
    axes[0, 1].imshow(a[:, iter].reshape(new_size ))
    axes[0, 1].set_title('Target Frame')
    
    axes[1, 0].imshow(zptB[:, iter].reshape(new_size))
    axes[1, 0].set_title(f'dHVA Output Frame {iter}')
    
    axes[1, 1].imshow(a[:, iter].reshape(new_size ))
    axes[1, 1].set_title('Target Frame')
    # Add spacing between subplots to avoid overlap
    fig.tight_layout()
    
    
    