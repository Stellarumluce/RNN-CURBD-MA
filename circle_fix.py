# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:51:34 2024

@author: esnow
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Parameters
N = 100
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
T = 1440
dt = 0.1
tau = 1
t_vec = np.linspace(0, T, int(T/dt))
# Initialize variables
W = np.zeros((N, N))  # for each element of 1600 we have a designated w vector
J = g * np.random.randn(N, N) / math.sqrt(N)
w_f = 2.0 * (np.random.rand(N) - 0.5)  # fixed weight after init
x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn()
z = np.zeros((N,))  # Initialize z as a vector for 2D output

# Create the concentric circles pattern
sqrt_N = int(np.sqrt(N))
x = np.linspace(-1, 1, sqrt_N)
y = np.linspace(-1, 1, sqrt_N)
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2 + yy**2)
circles_pattern = (np.sin(4 * np.pi * r) > 0).astype(int)

# Flatten the pattern and tile it for each time step
circles_pattern_flat = circles_pattern.flatten()
circles_pattern_time = np.tile(circles_pattern_flat, (len(t_vec), 1)).T

# Introduce a slow temporal modulation to the circles pattern
modulation = 0.5 * (1 + np.sin(2 * np.pi * 0.1 * t_vec))  # Slow varying sinusoidal modulation
circles_pattern_time = circles_pattern_time * modulation

# Reshape the pattern to match the training data format
a = circles_pattern_time + 0.5
# Training Loop only time loop
for neuron_num in range(N):
    x = np.zeros((N, len(t_vec)))
    x[:, 0] = x_0
    r = np.tanh(x_0)
    ps_vec = []
    P = np.eye(N) / alpha  # Initialize learning update matrix P
    z = z_0
    
    for t in range(1, len(t_vec)):
        # Update neuron state
        x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J, r) + z * w_f)
        # Update firing rate
        r = np.tanh(x[:, t])  # Ensure r is (N,)
        
        # Update network output
        z = np.dot(W[:, neuron_num].T, r)  # Ensure z is a scalar 
        
        # Compute error
        e_minus = z - a[neuron_num, t-1]
        
        # RLS update
        k = np.dot(P, r)
        rPr = np.dot(r.T, k)
        c = 1.0 / (1.0 + rPr)
        W[:, neuron_num] = W[:, neuron_num] - e_minus * k * c
        P = P - c * np.outer(k, k)
        
        # Post-error calculation
        e_plus = np.dot(W[:, neuron_num].T, r) - a[neuron_num, t-1]
    
        if e_minus != 0:
            ps_val = e_plus / e_minus
        else:
            ps_val = 0  # Handle the division by zero case
    
        ps_vec.append(ps_val)

# Testing Phase
simtime_len = len(t_vec)
final_zpt = np.zeros((N, simtime_len))
for neuron_num in range(N):
    zpt = np.zeros(simtime_len)
    x_test = x[:, -1]  # Start from the last state of the training phase
    r_test = np.tanh(x_test)
    z = z_0
    for ti in range(simtime_len):
        x_test = (1 - dt / tau) * x_test + (dt / tau) * (np.dot(J, r_test) + z * w_f)
        r_test = np.tanh(x_test)
        z = np.dot(W[:, neuron_num].T, r_test)  # Use the final trained weights
        zpt[ti] = z
    final_zpt[neuron_num, :] = zpt.T

# Calculate Mean Absolute Error (MAE)
error_avg = np.mean(np.abs(final_zpt - a))
print(f'Testing MAE: {error_avg:.3f}')

for iter in range(280):
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

plt.figure()
plt.plot(range(1, len(ps_vec)+1), ps_vec)
plt.title('e+/e- training process')
plt.xlabel('Iteration')
plt.ylabel('e+/e-')
plt.show()
