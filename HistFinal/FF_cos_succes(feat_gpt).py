# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:34:17 2024

@author: esnow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random

# Set random seed for reproducibility
np.random.seed(5)

# Parameters
N = 100
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
T = 1440
dt = 0.5
tau = 1
t_vec = np.linspace(0, T, int(T/dt))
W = np.zeros((N, N)) # for each element of 1600 we have a designated w vector
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
J_N = np.zeros((N, N, N))
J_pre_N = np.zeros((N, N, N))
W_pre = np.zeros((N, N))    
# Generate x positions
x_positions = np.linspace(0, 2 * np.pi, N)  # x positions

# Create the 2D wave function z = cos(x + t)
a = np.zeros((N, len(t_vec)))  # Initialize the wave matrix

for i, t in enumerate(t_vec):
    a[:, i] = np.cos(x_positions + t) 

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
            x_d[:, t] = (1 - dt / tau) * x_d[:, t-1] + (dt / tau) * (np.dot(J_d, r_d) + 2*u * a[neuron_num,t-1])
            x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J_N[:,:,neuron_num], r) + u * a[neuron_num,t-1])
            # Update firing rate
            r_d = np.tanh(x_d[:, t])  # Ensure r_d is (N,)
            r = np.tanh(x[:, t])  # Ensure r is (N,)
            
            # Compute error for recurrent weights
            e = np.dot(J_N[:,:,neuron_num], r) - np.dot(J_d, r_d) - u * a[neuron_num,t-1]
        
            k = np.dot(P, r)
            rPr = np.dot(r.T, k)
            c = 1.0 / (1.0 + rPr)
        
            J_N[:,:,neuron_num] = J_N[:,:,neuron_num] - np.outer(e, k) * c
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
final_zpt = np.zeros((N, simtime_len))
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
print(f'Testing MAE: {error_avg:.3f}')

for iter in range(200):
    # Plot the results
    plt.figure()
    plt.subplot(1, 2, 1)
    img1 = plt.imshow(final_zpt[:, iter].reshape(10, 10), aspect='auto')
    plt.title('Output Frame')
    plt.colorbar(img1)

    plt.subplot(1, 2, 2)
    img2 = plt.imshow(a[:, iter].reshape(10, 10), aspect='auto')
    plt.title('Target Frame')
    plt.colorbar(img2)
    plt.show()

