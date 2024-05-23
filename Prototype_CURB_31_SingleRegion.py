# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:42:17 2024

@author: esnow
"""

import numpy as np
import matplotlib.pyplot as plt
# Function to simulate the neural network dynamics for one iteration
def simulate_network(N, T, g, dt, a, J):
    mean = 0
    std = g / np.sqrt(N)
    tau_WN = 0.1
    t_vec = np.linspace(0, T, int(T/dt))
    
    # Initialization
    x = np.zeros((N, len(t_vec)))  # Neuron activities
    h = np.zeros(len(t_vec))       # External input
    x[:, 0] = np.squeeze(np.random.normal(mean, 1, (N, 1)))  # Initial neural states
    FRate_Matrix = np.zeros((N, len(t_vec)))  # Neuron Firing Rate Matrix
    FRate_Matrix[:,0] =  np.tanh(x[:, 0])
    #Q = np.outer(x[:, 0], x[:, 0])
    #P = np.linalg.inv(Q)  # Initial inverse cross-correlation matrix
    P0 = 9
    h0 = 1
    h[0] = np.random.normal(mean, 1, (1, 1))
    P = P0*np.eye(N, dtype=int)
    # Calculate h dynamics
    for t in range(1, len(t_vec)):
        h[t] = (1 - dt/tau_WN) * h[t - 1] + dt * np.random.normal(mean, 1) / tau_WN
    
    # Simulate network dynamics and update weights
    for t in range(1, len(t_vec)):
        # Update neural activities
        x[:, t] = (1 - dt) * x[:, t - 1] + dt * np.dot(J[:, :, t - 1], np.tanh(x[:, t - 1])) + h[t]
        
        # Compute firing rates
        phi_pre = np.tanh(x[:, t - 1])
        phi = np.tanh(x[:, t])
        
        # Calculate error
        err = phi - a[:, t - 1]
        
        # Compute the scaling factor c
        c = 1 / (1 + np.dot(phi_pre.T, np.dot(P, phi_pre)))
        
        # Update the matrix P
        P = P - np.dot(np.dot(P, np.outer(phi, phi)), P) / (1 + np.dot(phi.T, np.dot(P, phi)))
        
        # Compute weight update dJ
        dJ = c * np.outer(np.dot(P, phi), err)
        
        # Update the weight matrix J
        J[:, :, t] = J[:, :, t - 1] + dJ
        
        FRate_Matrix[:,t] =  phi
    
    return FRate_Matrix, J

# Function to train the network over multiple iterations
def train_network(N, T, g, dt, a, num_iterations):
    # Initialize the weight matrix
    std = g / np.sqrt(N)
    J_init = std * np.random.normal(0, std, (N, N))
    
    # Create a 3D J matrix to store weights over time
    J = np.zeros((N, N, int(T/dt)))
    J[:, :, 0] = J_init
    Pval = []
    for iteration in range(num_iterations):
        FRate_Matrix, J = simulate_network(N, T, g, dt, a, J)
        mean_a = np.mean(a, axis=1, keepdims= True)
        explained_var = np.sum((a - FRate_Matrix)**2, axis=None)
        total_var = np.sum((a - mean_a)**2, axis=None)
        pval  = 1 - explained_var/total_var
        Pval.append(pval)
        print(f"Iteration {iteration + 1}/{num_iterations} completed")
        
    return FRate_Matrix, J, Pval

# Example usage
N = 200  # Number of neurons
T = 10.0  # Total time
g = 1.5  # Scaling factor for weight initialization
dt = 0.01  # Time step size
num_iterations = 160  # Number of training iterations
# Generate target data with a simple pattern (e.g., sinusoidal)
time_steps = np.linspace(0, T, int(T/dt))
a = np.sin(np.outer(np.linspace(0, 2 * np.pi, N), time_steps)) # Example target data


FRate_Matrix, trained_J, Pval = train_network(N, T, g, dt, a, num_iterations)
#print(trained_J)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), Pval, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Proportion of Variance Explained (pVar)')
plt.title('Training Progress')
plt.grid(True)
plt.show()
