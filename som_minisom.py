# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:46:48 2023

@author: inesmpsoares
"""

def what_to_cluster(data_frame, cluster_by):
    
    """
    Define the data frame to use for clustering based on the cluster_by parameter.

    Parameters:
        data_frame: A pandas DataFrame object containing the data to be clustered.
        cluster_by: A string specifying whether to cluster by 'gene' or 'sample'.

    Returns:
        data_cluster: A pandas DataFrame object representing the data frame to be used for clustering.
    """
    if cluster_by == 'gene':
        data_cluster = data_frame
        # If cluster_by is 'gene', the input data_frame is assigned directly to the data_cluster variable.
        
    if cluster_by == 'sample':
        data_cluster = data_frame.transpose(copy=True)
        # If cluster_by is 'sample', the data_frame is transposed (rows become columns) and the resulting transposed data frame is assigned to the data_cluster variable.

    else:
        print ('error = cluster_by must be gene or sample')
        # If cluster_by has any other value, an error message is printed.
    
    return data_cluster

#MiniSom

from minisom import MiniSom
from sklearn.datasets import load_breast_cancer
import time
import pandas as pd
import numpy as np

# set hyperparameters

som_grid_rows= 30
som_grid_columns=30
iterations=100
sigma=1
learning_rate = 0.5

#load data
filepath = 'C:/Users/inesm/Documents/Tese/Datasets/logCPM_centered.csv'
case_study1 = pd.read_csv(filepath, delimiter=",", header=0, index_col=0)
cs1 = what_to_cluster(case_study1, 'gene')

#som training

# Convert the DataFrame to a NumPy array
CS1 = cs1.to_numpy()
#data= load_breast_cancer(True)

#initialization

som=MiniSom(x=som_grid_rows, y=som_grid_columns, 
            input_len=CS1.shape[1], sigma= sigma, 
            learning_rate=learning_rate)

som.random_weights_init(CS1)

# training

start_time = time.time()
som.train_random(CS1, iterations) #training with 100 iterations
elapsed_time =time.time() - start_time

print(elapsed_time, "seconds")

# Errors SOM
quantization_error = som.quantization_error(CS1)
print(quantization_error)

#falta topographic
import numpy as np
from minisom import MiniSom

def calculate_topographic_error(som, CS1):
    topographic_error = 0
    for i in range(CS1.shape[0]):
        bmu = som.winner(CS1[i])
        neighbors = som.neighborhood(bmu, sigma=1.0)
        
        # Calculate the distance between the input data and the BMU's weight vector
        bmu_weight = som.get_weights()[bmu]
        distance_to_bmu = np.linalg.norm(CS1[i] - bmu_weight)
        
        # Check if the distance exceeds a threshold (e.g., 1.0)
        if distance_to_bmu > 1.0:
            topographic_error += 1  # No neighbor in the input space
    return topographic_error / CS1.shape[0]


topographic_error = calculate_topographic_error(som, CS1)
print("Topographic Error:", topographic_error)


from pylab import plot, axis, show, pcolor, colorbar, bone
bone()
pcolor(som.distance_map().T, cmap='Spectral_r') #distance map as background
colorbar()
