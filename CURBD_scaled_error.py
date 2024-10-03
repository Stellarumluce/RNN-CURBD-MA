# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:26:34 2024

@author: esnow
"""

import math
import random
import numpy as np
import numpy.random as npr
import numpy.linalg
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.sparse import random as sparse_random
from skimage.transform import resize

# Function to load retinal wave data
def load_retinal_wave_data(wave_directory):
    wave_files = [os.path.join(wave_directory, f) for f in os.listdir(wave_directory) if f.endswith('.npy')]
    waves = [np.load(wave_file) for wave_file in wave_files]
    return waves

# Function to resize data
def resize_data(data, new_size):
    T = data.shape[2]
    resized_data = np.zeros((new_size[0], new_size[1], T))
    for t in range(T):
        resized_frame = resize(data[:, :, t], new_size, anti_aliasing=True, mode='reflect')
        threshold = 0.01
        resized_data[:, :, t] = (resized_frame > threshold).astype(int)
    return resized_data

# Function to normalize data
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Function to initialize variables
def initialize_variables(N, g, p, alpha):
    scale = 1.0 / np.sqrt(p * N)
    sparse_matrix = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn)
    J_d = sparse_matrix.toarray() * g * scale
    u_in = np.random.uniform(-1, 1, size=(N))
    u = np.random.uniform(-1, 1, size=(N))
    P = np.eye(N) / alpha
    x_0 = 0.5 * np.random.randn(N)
    return J_d, u_in, u, P, x_0

# Function to downsample data
def downsample_data(data, original_dt, new_dt):
    factor = int(new_dt / original_dt)
    return data[:, ::factor]

# Load Retinal Wave Data
wave_directory = 'data/generated_waves/Trial0_Directional/numpy_waves/'
waves = load_retinal_wave_data(wave_directory)
retinal_data = waves[1]  # Using the fifth wave file

# Determine Frame Dimensions
frame_height, frame_width, T = retinal_data.shape
print(f"Frame dimensions: {frame_height}x{frame_width}, Number of frames: {T}")

# Load the excitatory data
exc_data = np.load('D:/MA/activity_code/activity_code/data/rates_exc_Htriggered_Hampdirect_trial0.npy')
inh_data = np.load('D:/MA/activity_code/activity_code/data/rates_inh_Htriggered_Hampdirect_trial0.npy')

# Downsample the excitatory data to match the retinal data's temporal resolution
original_dt = 1  # Original temporal resolution of exc_data is 1 ms
new_dt = 10  # New temporal resolution should match retinal data, which is 0.5 seconds (500 ms)
exc_data_downsampled = downsample_data(exc_data, original_dt, new_dt)

# Resizing the excitatory data frames from 40x40 to 20x20
new_size = (20, 20)
T_resized = retinal_data.shape[2]
resized_exc_data = np.zeros((new_size[0] * new_size[1], T_resized))

for t in range(T_resized):
    frame = exc_data_downsampled[:, t].reshape(40, 40)  # Assuming original size of exc_data frames is 40x40
    resized_frame = resize(frame, new_size, anti_aliasing=True, mode='reflect')
    resized_exc_data[:, t] = resized_frame.reshape(new_size[0] * new_size[1])

print(f"Resized excitatory data dimensions: {resized_exc_data.shape}")

# Resizing the retinal data frames from 64x64 to 20x20
resized_retinal_data = resize_data(retinal_data, new_size)
print(f"Resized retinal data dimensions: {resized_retinal_data.shape}")

# Normalize the data
normalized_exc_data = normalize_data(resized_exc_data)
normalized_retinal_data = normalize_data(resized_retinal_data)

# Vectorize retinal data
vectorized_retinal_data = normalized_retinal_data.reshape(new_size[0] * new_size[1], T_resized)
# Assuming vectorized_retinal_data is already defined and has shape (N, T_resized)
N, T_resized = vectorized_retinal_data.shape
#N = new_size[0] * new_size[1]
T = T_resized
dt = 1
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
tau = 3 # 8/8.1 for 15,15 39/38.8 for 20x20 2.99 for /2 /4
# or 5 looks good 4 even better 3.7 is no longer good 3.85 okish 3.79 good with Lily's input 3.8 really good chi2 2.27 (10,10)
ampInWN=0.001
P0=1.0
tauWN= 0.1
num_iter = 2000
t_vec = np.linspace(0, T, int(T/dt))
# Generate x positions
x_positions = np.linspace(0, 2 * np.pi, N)  # x positions
a = normalized_exc_data
number_units = a.shape[0]
number_learn = a.shape[0]
learnList = npr.permutation(number_units)
iTarget = learnList[:number_learn]
ampWN = math.sqrt(tauWN / dt)
iWN = ampWN * npr.randn(number_units, len(t_vec))
inputWN = np.ones((number_units, len(t_vec)))
for tt in range(1, len(t_vec)):
    inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * np.exp(- (dt / tauWN))
inputWN = ampInWN * inputWN
u = np.random.uniform(-1, 1, size=(N))
# initialize directed interaction matrix J
J = g * npr.randn(number_units, number_units) / math.sqrt(number_units) # scaled
J0 = J.copy()
x = np.zeros((N, len(t_vec)))
# set up target training data
Adata = a.copy()
Adata = Adata / Adata.max()
Adata = np.minimum(Adata, 0.999)
Adata = np.maximum(Adata, -0.999)
stdData = np.std(Adata[iTarget, :])
# get indices for each sample of model data
# initialize some others
RNN = np.zeros((number_units, len(t_vec)))
chi2s = []
pVars = []

# initialize learning update matrix (see Sussillo and Abbot, 2009)
PJ = P0 * np.eye(number_learn)
# loop along training runs
for iter in range(0, num_iter): #iteration loop
    H = Adata[:, 0, np.newaxis] # init
    RNN[:, 0, np.newaxis] = H  # ugh
    # variables to track when to update the J matrix since the RNN and
    # data can have different dt values
    iLearn = 0  # keeps track of last data point learned
    chi2 = 0.0
    for tt in range(1, len(t_vec)): #temporal loop
        # compute next RNN step
        RNN[:, tt, np.newaxis] = H
        JR = (J.dot(RNN[:, tt]).reshape((number_units, 1)) +
              inputWN[:, tt, np.newaxis] + 0.01*u[:,np.newaxis] * vectorized_retinal_data[:, tt-1, np.newaxis])
        x[:, tt, np.newaxis] = x[:, tt-1,np.newaxis] + dt * (-x[:, tt-1, np.newaxis] + JR) / tau
        H = np.tanh(x[:, tt,np.newaxis])
        err = (RNN[:, tt, np.newaxis] - Adata[:, iLearn, np.newaxis])/np.sqrt(N/2) #N/2,4,5 9--10--20--100
        iLearn = iLearn + 1
        # update chi2 using this error
        chi2 += np.mean(err ** 2)

        if iter < num_iter:
            r_slice = RNN[iTarget, tt].reshape(number_learn, 1)
            k = PJ.dot(r_slice)
            rPr = (r_slice).T.dot(k)[0, 0]
            c = 1.0 / (1.0 + rPr)
            PJ = PJ - c * (k.dot(k.T))
            J[:, iTarget.flatten()] = J[:, iTarget.reshape((number_units))] - c * np.outer(err.flatten(),
                                                                                           k.flatten())

    rModelSample = RNN[iTarget, :]
    distance = np.linalg.norm(Adata[iTarget, :] - rModelSample)
    pVar = 1 - (distance / (math.sqrt(len(iTarget) * len(t_vec))
                            * stdData)) ** 2
    pVars.append(pVar)
    chi2s.append(chi2)
plt.figure()
plt.plot(pVars)
plt.ylabel('pVar')
plt.figure()
plt.plot(chi2s)
plt.ylabel('MSE')
idx = npr.choice(range(len(iTarget)))
plt.figure()
plt.plot(t_vec, RNN[iTarget[idx], :])
plt.plot(t_vec, Adata[iTarget[idx], :])
for iter in range(T_resized):
    # Plot the results
    plt.figure()
    plt.subplot(1, 2, 1)
    img1 = plt.imshow(RNN[:, iter].reshape(new_size), aspect='auto')
    plt.title(f'Output Frame {iter}')
    plt.colorbar(img1)

    plt.subplot(1, 2, 2)
    img2 = plt.imshow(a[:, iter].reshape(new_size), aspect='auto')
    plt.title(f'Target Frame {iter}')
    plt.colorbar(img2)
    
plt.show()