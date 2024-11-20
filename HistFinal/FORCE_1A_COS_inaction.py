# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:44:11 2024

@author: esnow
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from skimage.transform import resize
import math


# Determine Frame Dimensions
N = 1600
T = 144
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
dt = 1
tau = 1.25
t_vec = np.linspace(0, T, int(T/dt))
# Initialize variables
W = np.zeros((N, N)) # for each element of 1600 we have a destinated w vector
J = g * np.random.randn(N, N) / math.sqrt(N)
w_f = 2.0 * (np.random.rand(N) - 0.5)  # fixed weight after init
x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn()
z = np.zeros((N,))  # Initialize z as a vector for 2D output
x_positions = np.linspace(0, 2 * np.pi, N)  # x positions

# Create the 2D wave function z = cos(x + t)
a = np.zeros((N, len(t_vec)))  # Initialize the wave matrix

for i, t in enumerate(t_vec):
    a[:, i] = np.cos(x_positions + t) 
    
x = np.zeros((N, len(t_vec)))
x[:, 0] = x_0
r = np.tanh(x_0)
ps_vec = []
P = np.eye(N) / alpha  # Initialize learning update matrix P
z = z_0
for t in range(1, len(t_vec)):
    # Update neuron state retinal_data_flat_cycled
    # if t <= retinal_data.shape[2]:
    
    x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J, r) + z * w_f)
    # Update firing rate
    r = np.tanh(x[:, t])  # Ensure r is (N,)
    
    # Update network output
    z = np.dot(W.T , r) 
    
    # Compute error
    e_minus = z - a[:,t] # 
    
                                              
    # RLS update
    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = 1.0 / (1.0 + rPr)
    W -= c * np.outer(k, e_minus) #key training process #sensitive to errorc * np.outer(k, e)
    P = P - c * np.outer(k, k)
    
    # Post-error calculation
    e_plus = np.dot(W.T, r) - a[:,t]

# Testing Phase
simtime_len = len(t_vec)
final_zpt = np.zeros((N,simtime_len))
x_test = x[:, -1]  # Start from the last state of the training phase
r_test = np.tanh(x_test)
z = z_0
for ti in range(simtime_len):
    x_test = (1 - dt / tau) * x_test + (dt / tau) * (np.dot(J, r_test) + z * w_f)
    r_test = np.tanh(x_test)
    z = np.dot(W.T, r_test)  # Use the final trained weights
    final_zpt[:, ti] = z
# Calculate Mean Absolute Error (MAE)
  # Assuming ft2 is the same as the target function used for training
error_avg = np.mean(np.abs(final_zpt - a))
print(f'Testing MAE: {error_avg:.3f}')

for iter in range(simtime_len):
    # Plot the results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im_1 = plt.imshow(final_zpt[:, iter].reshape(40, 40))
    plt.colorbar(im_1)
    plt.title('Output Frame FORCE Fig 1A')
    plt.subplot(1, 2, 2)
    im_2 = plt.imshow(a[:, iter].reshape(40, 40))
    plt.colorbar(im_2)
    plt.title('Target Frame')
    plt.tight_layout()