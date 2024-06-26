# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:27:03 2024

@author: esnow
"""

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import math

# np.random.seed(14) #cycled 10 times

# Parameters
N = 1600
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 10.0
T = 1000
dt = 0.5
tau = 1
t_vec = np.linspace(0, T, int(T/dt))
M = np.eye(N) #To be discussed
# Initialize variables
W = np.zeros((N, len(t_vec)))
J = g * np.random.randn(N, N) / math.sqrt(N)
w_f = 2.0 * (np.random.rand(N) - 0.5)  # fixed weight after init
P = np.eye(N) / alpha  # Initialize learning update matrix P
x = np.zeros((N, len(t_vec)))
x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn(N)
z = np.zeros((N,))  # Initialize z as a vector for 2D output
x[:, 0] = x_0
r = np.tanh(x_0)
z = z_0
ps_vec = []
# Target function: sum of 4 sinusoids
# Generate x positions
x_positions = np.linspace(0, 2 * np.pi, N)  # x positions

# Create the 2D wave function z = cos(x + t)
a = np.zeros((N, len(t_vec)))  # Initialize the wave matrix

for i, t in enumerate(t_vec):
    a[:, i] = np.cos(x_positions + t)

# a = exc_data[:N, :int(T/dt)]  # Adjust size to match the RNN training duration
# Training Loop only time loop
for t in range(1, len(t_vec)):
    # Update neuron state retinal_data_flat_cycled
    # if t <= retinal_data.shape[2]:
    
    x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J, r) + z * w_f)
    # Update firing rate
    r = np.tanh(x[:, t])  # Ensure r is (N,)
    
    # Update network output
    z = W[:, t-1] * r  # Ensure z is a vector
    
    # Compute error
    e_minus = z - a[:,t-1] # 
    mse_minus = np.sum(np.square(e_minus)) / N
                                              
    # RLS update
    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = 1.0 / (1.0 + rPr)
    W[:, t] = W[:, t-1] - mse_minus * k * c #key training process #sensitive to error
    P = P - c * np.outer(k, k)
    
    # Post-error calculation
    e_plus = np.dot(W[:, t], r) - a[:,t-1]
    mse_plus = np.sum(np.square(e_plus)) / N
    if mse_minus != 0:
        ps_val = mse_plus / mse_minus
    else:
        ps_val = 0  # Handle the division by zero case

    ps_vec.append(ps_val)

# Testing Phase
simtime_len = len(t_vec)
zpt = np.zeros((N, simtime_len))
x_test = np.zeros((N, len(t_vec)))
x_test[:,0] = x[:, -1]  # Start from the first state of the training phase
r_test = np.tanh(x[:, 0]) # Init

for ti in range(simtime_len):
    x_test[:, ti] = (1 - dt / tau) * x_test[:, ti-1] + (dt / tau) * (np.dot(J, r_test) + z * w_f)
    r_test = np.tanh(x_test[:, ti])
    z_diff = W[:, -1] * r_test  # Use the final trained weights
    zpt[:,ti] = z_diff

# Calculate Mean Absolute Error (MAE)
# ft2 = ft  # Assuming ft2 is the same as the target function used for training
error_avg = np.mean(np.abs(zpt - a))
print(f'Testing MAE: {error_avg:.3f}')

# Plot the results
plt.plot(t_vec, np.mean(a, axis = 0 ), label='Target')
plt.plot(t_vec, np.mean(zpt, axis = 0 ), label='Output')
plt.xlabel('Time (s)')
plt.title('mean output comparison')
plt.figure()
plt.plot(range(1, len(ps_vec)+1), ps_vec)
plt.title('e+/e- training process')
plt.legend()
plt.figure()
plt.subplot(1,2,1)
plt.imshow(zpt[:, 500].reshape(40,40))
plt.subplot(1,2,2)
plt.imshow(a[:, 500].reshape(40,40))
plt.show()
