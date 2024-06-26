# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
import math

np.random.seed(14) 

# Parameters
N = 1600
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
T = 1440
dt = 0.5
tau = 1
t_vec = np.linspace(0, T, int(T/dt))
# Initialize variables
W = np.zeros((N, N)) # for each element of 1600 we have a destinated w vector
J = g * np.random.randn(N, N) / math.sqrt(N)
w_f = 2.0 * (np.random.rand(N) - 0.5)  # fixed weight after init
x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn()
z = np.zeros((N,))  # Initialize z as a vector for 2D output
# Target function: sum of 4 sinusoids

# Create the 2D wave function z = cos(x + t)
a = np.zeros((N, len(t_vec)))  # Initialize the wave matrix

amp = 1.3
freq = 1 / 60
ft = (amp / 1.0 * np.sin(1 * np.pi * freq * t_vec) + 
      amp / 2.0 * np.sin(2 * np.pi * freq * t_vec) + 
      amp / 6.0 * np.sin(3 * np.pi * freq * t_vec) +
      amp / 3.0 * np.sin(4 * np.pi * freq * t_vec))  # Example target data could be more complex

a = ft / 1.5
a = np.tile(a, (N,1))
# a = exc_data[:N, :int(T/dt)]  # Adjust size to match the RNN training duration
# Training Loop only time loop
for neuron_num in range(100):
    # neuron_num = 10
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
        z = np.dot(W[:, neuron_num].T , r ) # Ensure z is a scalar 
        
        # Compute error
        e_minus = z - a[neuron_num,t-1] # 
        
                                                  
        # RLS update
        k = np.dot(P, r)
        rPr = np.dot(r.T, k)
        c = 1.0 / (1.0 + rPr)
        W[:, neuron_num] = W[:, neuron_num] - e_minus * k * c #key training process #sensitive to error
        P = P - c * np.outer(k, k)
        
        # Post-error calculation
        e_plus = np.dot(W[:, neuron_num].T, r) - a[neuron_num,t-1]
    
        if e_minus != 0:
            ps_val = e_plus / e_minus
        else:
            ps_val = 0  # Handle the division by zero case
    
        ps_vec.append(ps_val)
# Testing Phase
simtime_len = len(t_vec)
final_zpt = np.zeros((N,simtime_len))
for neuron_num in range(100):
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
  # Assuming ft2 is the same as the target function used for training
error_avg = np.mean(np.abs(final_zpt - a))
print(f'Testing MAE: {error_avg:.3f}')

for iter in range(100):
    # Plot the results
    plt.figure()
    plt.plot(t_vec, a[iter,:].T, label='Target')
    plt.plot(t_vec, final_zpt[iter,:].T, label='Output')
plt.figure()
plt.plot(range(1, len(ps_vec)+1), ps_vec)
plt.title('e+/e- training process')
plt.legend()
plt.show()
