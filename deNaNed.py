# -*- coding: utf-8 -*-
"""
Created on Tue May 28 23:39:30 2024

@author: esnow
"""


import numpy as np
import matplotlib.pyplot as plt
import math
import random
# Compared with the paper's implementation the nans are for now to be hopefully solved by permutation of the learning neurons
def simulate_network(N, T, g, dt, a, J, i_target):

    tau_WN = 0.1
    tau_RNN = 0.01
    t_vec = np.linspace(0, T, int(T/dt))
    
    # Initialization
    x = np.zeros((N, len(t_vec)))  # Neuron activities
    h = np.zeros(len(t_vec))       # External input
    x[:, 0] = a[:, 0]  # Initial neural states question
    
    # maybe try to set up a permutated learn_list #
    
    FRate_Matrix = np.zeros((N, len(t_vec)))  # Neuron Firing Rate Matrix
    FRate_Matrix[:,0] = np.tanh(x[:, 0]) #RNN[:, 0, np.newaxis] = nonLinearity(H) anchor point line 136
    P0 = 1.0
    P = P0 * np.eye(N)  # Initialize learning update matrix P
    
    ampInWN=0.01
    ampWN = math.sqrt(tau_WN/dt)
    iWN = ampWN * np.random.randn(N, len(t_vec))
    inputWN = np.ones((N, len(t_vec)))
    for tt in range(1, len(t_vec)):
        inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt-1] - iWN[:, tt])*np.exp(- (dt / tau_WN))
    h = ampInWN * inputWN
    #checked
    
    # Simulate network dynamics and update weights
    # Check For Time Alignment!
    for t in range(1, len(t_vec)): 
        FRate_Matrix[:,t] = np.tanh(x[i_target, t-1]) #phi pre
        x[:, t] = (1 - dt/tau_RNN) * x[:, t-1] + dt / tau_RNN * (np.dot(J[:, :, t-1], FRate_Matrix[:,t]) + h[:,t]) #checked
        err = FRate_Matrix[:,t] - a[:, t-1] #?
        r_slice = FRate_Matrix[i_target, tt]
        k = np.dot(P, r_slice)
        rPr = np.dot(r_slice.T, k)
        c = 1.0 / (1.0 + rPr)
        P = P - c*(np.outer(k,k)) # prob
        dJ = c * np.outer(err.flatten(), k.flatten())
        J[:, i_target, t] = J[:, i_target, t-1] - dJ
         
    
    return FRate_Matrix, J

def train_network(N, T, g, dt, a, num_iterations, i_target):
    #std = g / np.sqrt(N)
    J_init = g * np.random.randn(N, N) / math.sqrt(N) #check
    J = np.zeros((N, N, int(T/dt)))
    J[:, :, 0] = J_init
    Pvar = []
    t_vec = np.linspace(0, T, int(T/dt))
    a_std = np.std(a[i_target,:]) #scalar even sloppier of total std
    for iteration in range(num_iterations):
        FRate_Matrix, J_updated = simulate_network(N, T, g, dt, a, J, i_target)
        J = J_updated
        rModelSample = FRate_Matrix[i_target, :]
        # RSS = np.sum((a[i_target, :] - rModelSample) ** 2)
        # TSS = np.sum((a[i_target, :] - np.mean(a[i_target, :], axis=1, keepdims=True)) ** 2)
        distance = np.linalg.norm(a[i_target, :] - rModelSample)
        pVar = 1 - (distance / (math.sqrt(len(i_target) * len(t_vec))
                    * a_std)) ** 2
        #pVar = 1 - (RSS / TSS)
        Pvar.append(pVar)
        print(f"Iteration {iteration + 1}/{num_iterations} completed")
    return FRate_Matrix, J, Pvar

# Example usage
N = 100  # Number of neurons
T = 1.0  # Total time
g = 1.5  # Scaling factor for weight initialization
dt = 0.01  # Time step size
num_iterations = 120  # Number of training iterations
frequency = 1  # Frequency of the sinusoid in Hz
time_steps = np.linspace(0, T, int(T/dt))
a = np.sin(2 * np.pi * frequency * time_steps) # Example target data
a = np.tile(a, (N, 1)) # MIGHT BE A PROBLEM!! why dn't we make it one dim
a = a/a.max()
a = np.minimum(a, 0.999)
a = np.maximum(a, -0.999) # 
learn_list = np.random.permutation(N) #random vector from 0 - N-1 
#(The permutation ensures that the order in which the units are selected for learning is randomized. 
#This is important for breaking any inherent order or bias in the data, promoting more robust and generalized learning.)
N_learn = a.shape[0]
i_target = learn_list[:N_learn] #picking the learning/ target neurons
FRate_Matrix, trained_J, Pvar = train_network(N, T, g, dt, a, num_iterations, i_target)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), Pvar, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Proportion of Variance Explained (pVar)')
plt.title('Training Progress')
plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(time_steps,FRate_Matrix[57,:])
plt.xlabel('Time')
plt.ylabel('Firing Rate')
plt.show()
