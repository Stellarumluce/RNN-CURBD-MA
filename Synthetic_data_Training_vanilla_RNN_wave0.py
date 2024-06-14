# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:47:18 2024

@author: esnow
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import math

np.random.seed(14)
# Load Retinal Wave Data
wave_directory = 'data/generated_waves/Trial0_Directional/numpy_waves/'
wave_files = [os.path.join(wave_directory, f) for f in os.listdir(wave_directory) if f.endswith('.npy')]
waves = [np.load(wave_file) for wave_file in wave_files]
retinal_data = waves[0]  # Assuming using the first wave files

# Determine Frame Dimensions
frame_height, frame_width, T = retinal_data.shape
N = frame_height * frame_width
print(f"Frame dimensions: {frame_height}x{frame_width}, Number of frames: {T}, Network size (N): {N}")
# Compute the average frame across all time steps across all frames
average_frame = np.mean(retinal_data, axis=2)
retinal_flattened = average_frame.flatten()

# Load the data from the .npy files
exc_data = np.load('D:/MA/activity_code/activity_code/data/rates_exc_Htriggered_Hampdirect_trial0.npy')
inh_data = np.load('D:/MA/activity_code/activity_code/data/rates_inh_Htriggered_Hampdirect_trial0.npy')


x_0 = retinal_flattened
# Parameters
N = retinal_flattened.shape[0]
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
T = 144
dt = 0.1
tau = 1
t_vec = np.linspace(0, T, int(T/dt))

# Initialize variables
W = np.zeros((N, len(t_vec)))
J = g * np.random.randn(N, N) / math.sqrt(N)
w_f = 2.0 * (np.random.rand(N) - 0.5)  # fixed weight after init
P = np.eye(N) / alpha  # Initialize learning update matrix P
x = np.zeros((N, len(t_vec)))
# x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn()

x[:, 0] = x_0
r = np.tanh(x_0)
z = z_0
ps_vec = []

a = exc_data[:N, :int(T/dt)]  # Adjust size to match the RNN training duration
a = np.mean(a, axis=0)
# Training Loop only time loop
for t in range(1, len(t_vec)):
    # Update neuron state
    x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J, r) + z * w_f)
    
    # Update firing rate
    r = np.tanh(x[:, t])  # Ensure r is (N,)
    
    # Update network output
    z = np.dot(W[:, t-1].T, r)  # Ensure z is a scalar
    
    # Compute error
    e_minus = z - a[t-1]
    
    # RLS update
    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = 1.0 / (1.0 + rPr)
    W[:, t] = W[:, t-1] - e_minus * k * c #key training process
    P = P - c * np.outer(k, k)
    
    # Post-error calculation
    e_plus = np.dot(W[:, t].T, r) - a[t-1]
    if e_minus != 0:
        ps_val = e_plus / e_minus
    else:
        ps_val = 0  # Handle the division by zero case

    ps_vec.append(ps_val)

# Testing Phase
simtime_len = len(t_vec)
zpt = np.zeros(simtime_len)
x_test = x[:, 0]  # Start from the last state of the training phase
r_test = np.tanh(x_test)

for ti in range(simtime_len):
    x_test = (1.0 - dt) * x_test + np.dot(J, r_test * dt) + w_f * (z * dt)
    r_test = np.tanh(x_test)
    z = np.dot(W[:, -1].T, r_test)  # Use the final trained weights
    zpt[ti] = z

# Calculate Mean Absolute Error (MAE)
# ft2 = ft  # Assuming ft2 is the same as the target function used for training
error_avg = np.mean(np.abs(zpt - a))
print(f'Testing MAE: {error_avg:.3f}')

# Plot the results
plt.plot(t_vec, a, label='Target')
plt.plot(t_vec, zpt, label='Output')
plt.figure()
plt.plot(range(1, len(ps_vec)+1), ps_vec)
plt.title('e+/e- training process')
plt.legend()
plt.show()
