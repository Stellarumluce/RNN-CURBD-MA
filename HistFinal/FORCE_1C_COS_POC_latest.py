# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:57:10 2024

@author: esnow
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix

# Set random seed for reproducibility

# Parameters
N = 1600
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
T = 144
dt = 0.5
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

# Generate x positions
x_positions = np.linspace(0, 2 * np.pi, N)  # x positions

# Create the 2D wave function z = cos(x + t)
a = np.zeros((N, len(t_vec)))  # Initialize the wave matrix

for i, t in enumerate(t_vec):
    a[:, i] = np.cos(x_positions + t) 
# a = exc_data[:N, :int(T/dt)]  # Adjust size to match the RNN training duration
# Training Loop only time loop

# Training Loop only time loop
for t in range(1, len(t_vec)):
    # Update neuron state
    x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J, r))
    
    # Update firing rate
    r = np.tanh(x[:, t])  # Ensure r is (N,)
    
    # Update network output
    z = np.dot(W.T, r)  # Ensure z is a scalar
    
    # Compute error
    e_minus = z - a[:,t-1]
    
    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = 1.0 / (1.0 + rPr)
    W -= c * np.outer(k, e_minus) #key training process #sensitive to errorc * np.outer(k, e)
    P = P - c * np.outer(k, k)
   
 

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
    # Plot the results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im_1 = plt.imshow(zpt[:, iter].reshape(40, 40))
    plt.colorbar(im_1)
    plt.title('Output Frame FORCE Fig 1C')
    plt.subplot(1, 2, 2)
    im_2 = plt.imshow(a[:, iter].reshape(40, 40))
    plt.colorbar(im_2)
    plt.title('Target Frame')
    plt.tight_layout()