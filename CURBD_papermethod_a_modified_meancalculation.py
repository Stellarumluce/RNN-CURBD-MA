# -*- coding: utf-8 -*-
"""
Created on Thu May 23 23:53:00 2024

@author: esnow
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

def simulate_network(N, T, g, dt, a, J):
    mean = 0
    std = g / np.sqrt(N)
    tau_WN = 0.1
    tau_RNN = 0.01
    t_vec = np.linspace(0, T, int(T/dt))
    
    # Initialization
    x = np.zeros((N, len(t_vec)))  # Neuron activities
    h = np.zeros(len(t_vec))       # External input
    x[:, 0] = np.squeeze(np.random.normal(mean, 1, (N, 1)))  # Initial neural states
    FRate_Matrix = np.zeros((N, len(t_vec)))  # Neuron Firing Rate Matrix
    FRate_Matrix[:,0] = np.tanh(x[:, 0])
    P0 = 1
    h0 = 1
    ampInWN=0.01
    P = P0 * np.eye(N)  # Initialize learning update matrix P
    
    ampWN = math.sqrt(tau_WN/dt)
    iWN = ampWN * np.random.randn(N, len(t_vec))
    inputWN = np.ones((N, len(t_vec)))
    for tt in range(0, len(t_vec)-1):
        inputWN[:, tt+1] = iWN[:, tt+1] + (inputWN[:, tt] - iWN[:, tt+1])*np.exp(- (dt / tau_WN))
    h = ampInWN * inputWN

    
    # Simulate network dynamics and update weights
    for t in range(0, len(t_vec)-1): #change the range t -> t+1
        x[:, t+1] = (1 - dt/tau_RNN) * x[:, t] + dt / tau_RNN * (np.dot(J[:, :, t], np.tanh(x[:, t])) + h[:,t]) 
        phi_pre = np.tanh(x[:, t])
        
        err = phi_pre - a[:, t] #?
        k = np.dot(P, phi_pre)
        rPr = np.dot(phi_pre.T, k)
        c = 1 / (1 + rPr)
        P = P - c*(k.dot(k.T))
        dJ = c * np.outer(err, k)
        J[:, :, t+1] = J[:, :, t] + dJ
        FRate_Matrix[:,t] = phi_pre
    
    return FRate_Matrix, J

def train_network(N, T, g, dt, a, num_iterations):
    #std = g / np.sqrt(N)
    J_init = g * np.random.randn(N, N) / math.sqrt(N)
    J = np.zeros((N, N, int(T/dt)))
    J[:, :, 0] = J_init
    Pvar = []
    for iteration in range(num_iterations):
        FRate_Matrix, J_updated = simulate_network(N, T, g, dt, a, J)
        J = J_updated
        #mean_a = np.mean(a, axis=1, keepdims=True)
        mean_temp = np.mean(a, axis=0, keepdims= True) #different mean through time steps
        mean_a = np.tile(mean_temp, (N, 1))
        explained_var = np.sum((a - FRate_Matrix)**2, axis=None)
        total_var = np.sum((a - mean_a)**2, axis=None)
        pvar = 1 - explained_var / total_var
        Pvar.append(pvar)
        print(f"Iteration {iteration + 1}/{num_iterations} completed")
    return FRate_Matrix, J, Pvar

# Example usage
N = 100  # Number of neurons
T = 5.0  # Total time
g = 1.5  # Scaling factor for weight initialization
dt = 0.01  # Time step size
num_iterations = 120  # Number of training iterations
frequency = 1  # Frequency of the sinusoid in Hz
time_steps = np.linspace(0, T, int(T/dt))
a = np.sin(2 * np.pi * frequency * time_steps) # Example target data
a = np.tile(a, (N, 1))
FRate_Matrix, trained_J, Pvar = train_network(N, T, g, dt, a, num_iterations)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), Pvar, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Proportion of Variance Explained (pVar)')
plt.title('Training Progress')
plt.grid(True)
plt.plot(time_steps,FRate_Matrix[57,:])
plt.show()
