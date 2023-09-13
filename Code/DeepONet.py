"""
Created August 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

USAGE: 
    For training a DeepONet for predicting EXAFS from XANES
    
Code structure from: 
    https://github.com/PredictiveIntelligenceLab/ImprovedDeepONets
"""

import os

#import numpy as np
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import jax
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import selu
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import pandas as pd


######################################################################
#######################  Standard DeepONets ##########################
######################################################################

# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, fft, w,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        self.fft = fft
        self.w = w
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        y = self.y[idx,:,:]
        u = self.u[idx,:,:]
        fft = self.fft[idx,:,:]
        w = self.w[idx,:,:]
        # Construct batch
        inputs = (u, y, w)
        outputs = (s, fft)
        return inputs, outputs




def modified_deeponet(branch_layers, trunk_layers, activation=selu):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b
    def init(rng_key1, rng_key2):
        U1, b1 = xavier_init_j(random.PRNGKey(12345), branch_layers[0], branch_layers[1])
        U2, b2 = xavier_init_j(random.PRNGKey(54321), trunk_layers[0], trunk_layers[1])
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        key2, *keys2 = random.split(rng_key2, len(trunk_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        trunk_params = list(map(init_layer, keys2, trunk_layers[:-1], trunk_layers[1:]))
        return (branch_params, trunk_params, U1, b1, U2, b2)
        
    def apply(params, u, y):
        branch_params, trunk_params, U1, b1, U2, b2 = params
    
        U = activation(np.dot(u, U1) + b1)
        V = activation(np.dot(y, U2) + b2)
        for k in range(len(branch_layers)-2):
            W_b, b_b = branch_params[k]
            W_t, b_t = trunk_params[k]
            
            B = activation(np.dot(u, W_b) + b_b)
            T = activation(np.dot(y, W_t) + b_t)
            u = B
            y = T
            
            u = np.multiply(B, U) + np.multiply(1 - B, V) 
            y = np.multiply(T, U) + np.multiply(1 - T, V) 
        
        W_b, b_b = branch_params[-1]
        W_t, b_t = trunk_params[-1]
        B = np.dot(u, W_b) + b_b
        T = np.dot(y, W_t) + b_t

        y_branch0, y_branch1 = np.split(B, 2, axis=1)
        y_trunk0,  y_trunk1 = np.split(T, 2, axis=1)     
        outputs_0 = np.sum(y_branch0 * y_trunk0, axis=1)
        outputs_1 = np.sum(y_branch1 * y_trunk1, axis=1)
        outputs = np.stack([outputs_0, outputs_1], axis=1)
        outputs = np.reshape(outputs,(outputs.shape[0], -1))
        return outputs

    return init, apply



class deepONet:
    
    # Initialize the class
    def __init__(self, layers_branch_low, layers_trunk_low): 

        if layers_branch_low[-1] != layers_trunk_low[-1]:
            raise ValueError("Output sizes of function NN and location NN do not match.")
        
        #Network initialization 
        self.init_low, self.apply_low = modified_deeponet(layers_branch_low, layers_trunk_low)
        params = self.init_low(random.PRNGKey(1), random.PRNGKey(2))

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(5e-5, 
                                                                      decay_steps=2000, 
                                                                       decay_rate=0.99))

        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # building loss function
        self.loss_training_log = []
        self.loss_testing_log  = []

    # =============================================
    # evaluation
    # =============================================
    # Loss function
    def loss(self, params, batch_f):
        params_f = params
        inputs_f, outputs_f = batch_f
        u_f, y_f, w_f = inputs_f
        s_f, fft_f = outputs_f


        s_pred_f = vmap(self.apply_low, (None, 0, 0))(params_f, u_f, y_f)
        t = ((w_f*fft_f).flatten()- (w_f*s_pred_f).flatten())**2
        loss_f = np.mean(t)
        
        mag1 = (s_pred_f[:, :, 0]**2 +s_pred_f[:, :, 1]**2 )
        mag2 = (fft_f[:, :, 0]**2 +fft_f[:, :, 1]**2)
        t = ((w_f*mag1).flatten()- (w_f*mag2).flatten())**2
        loss_mag = np.mean(t)

        loss = loss_f + loss_mag 
        return loss

    
    # Update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch_low):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params,batch_low)
        
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters
    def train(self, low_dataset, test_dataset, nIter = 10000):
        low_data = iter(low_dataset)
        test_data = iter(test_dataset)
        pbar = trange(nIter)
        
        # Main training loop
        for it in pbar:
            low_batch= next(low_data)
            test_batch = next(test_data)
            self.opt_state = self.step(next(self.itercount), self.opt_state,low_batch)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, low_batch)
                test_value = self.loss(params, test_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_testing_log.append(test_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value, 'test': test_value})

    # Evaluates predictions at test points  
    def predict_low(self, params, U_star, Y_star):
        params_low = params

        s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        return s_pred

                    
# =============================================
# =============================================
if __name__ == "__main__":
   
    #File where you want to save results
    save_directory = '../test/'
    
    #input.txt holds the training parameters for this run
    #it is stored in save_directory
    if not os.path.exists(save_directory):
        #If the save directory doesn't exist, make the directory and 
        #use default values
        os.makedirs(save_directory)
    if not os.path.isfile(save_directory + '/input.txt'):
        X_lim_max = 87
        X_lim_min = 0
        f  = open(save_directory + "/input.txt", 'w')
        f.write("%d \n" % X_lim_min)
        f.write("%d \n" % X_lim_max)
        f.close()
    else:
        with open(save_directory + '/input.txt') as f:
            lines = f.readlines()
        X_lim_max = int(lines[1].strip())
        X_lim_min = int(lines[0].strip())
            
            
    #Set some parameters used in training
    E_lim = 282
    X_lim = X_lim_max-X_lim_min
    N_fft = 40
    N_fft_x = 23
    batch_size = 25
    Ntrain = 2349
    Ntest = 245
    N = Ntrain + Ntest
    
    
    #DeepONet network size and training paramters
    Nnetwork = 200
    layers_branch = [X_lim,Nnetwork,Nnetwork,Nnetwork,Nnetwork,Nnetwork,Nnetwork]
    layers_trunk  = [2, Nnetwork, Nnetwork,Nnetwork,Nnetwork,Nnetwork,Nnetwork]
    
    epochs = 600000
 
    # ====================================
    data = scipy.io.loadmat('../Data/Copper_data_all.mat')
   
    X_x, X_y, E_x, E_y, ID, w  = (data["xval"].astype(np.float32), 
               data["yval"].astype(np.float32), 
               data["xval_out"].astype(np.float32),
                                 data["yval_out"].astype(np.float32),
                                 data["ID"].astype(np.float32),
                                 data["weight"].astype(np.float32))
    
    X_x = np.transpose(X_x)
    X_y = np.transpose(X_y)
    E_x = np.transpose(E_x)
    E_y = np.transpose(E_y)
    w = np.transpose(w)
    
    N_u = np.shape(X_x)[0]
    Train_range = np.arange(0, Ntrain)
    Test_range = np.arange(Ntrain, N_u)

    FFT_E = np.fft.rfft(E_y[:, 0:E_lim])[:, 0:N_fft]*4/E_lim
    r = np.real(FFT_E).reshape(N, N_fft, 1)
    im = np.imag(FFT_E).reshape(N, N_fft, 1)
    FFT_E = np.append(r, im, axis=2)

    y = np.vstack([np.arange(N_fft),np.arange(N_fft)])/N_fft
    x = np.tile(y.T, [N, 1, 1])
    
    
    #Set up forward data:
        #Branch: X_y
        #Trunk: E_x
        #Pred: E_y
        #FFT: FFT_E
    U_train_f = X_y[Train_range, X_lim_min:X_lim_max]
    X_train_f  = x[Train_range, :, :]
    S_train_f = E_y[Train_range, 0:E_lim]
    FFT_f = FFT_E[Train_range, :, :]
    w_f = w[Train_range, :]

    U_test_f = X_y[Test_range, X_lim_min:X_lim_max]
    X_test_f  = x[Test_range, :, :]
    S_test_f = E_y[Test_range, 0:E_lim]
    FFT_test_f = FFT_E[Test_range, :, :]
    w_test_f = w[Test_range, :]

    U_train_f = np.reshape(U_train_f,(-1,1,U_train_f.shape[1]))-1
    S_train_f = np.reshape(S_train_f,(-1,S_train_f.shape[1],1))
    w_f = np.reshape(w_f,(-1,w_f.shape[1],1))

    U_test_f = np.reshape(U_test_f,(-1,1,U_test_f.shape[1]))-1
    S_test_f = np.reshape(S_test_f,(-1,S_test_f.shape[1],1))
    w_test_f = np.reshape(w_test_f,(-1,w_test_f.shape[1],1))

    train_dataset_f = DataGenerator(U_train_f, X_train_f, S_train_f, FFT_f, w_f, batch_size)
    test_dataset_f = DataGenerator(U_test_f, X_test_f, S_test_f, FFT_test_f, w_test_f, batch_size)


    # ====================================
    # Train deeponet
    # ====================================

    model = deepONet(layers_branch, layers_trunk)
                    
    
    model.train(train_dataset_f, test_dataset_f, nIter=epochs)
    print('\n ... Training done ...')

    scipy.io.savemat(save_directory + "/losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'testing_loss':model.loss_testing_log})
    params = model.get_params(model.opt_state)
    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    np.save(save_directory + '/params.npy', flat_params)
            

    # ====================================
    # Save FEFF predictions
    # ====================================

    Npred = int(E_lim/2+1)
    S_pred_f = model.predict_low(params, U_test_f, X_test_f)

    pred_pad1 = np.concatenate((S_pred_f[:, :, 0].T, np.zeros([Ntest, 192]).T), axis=0).T
    pred_pad2 = np.concatenate((S_pred_f[:, :, 1].T, np.zeros([Ntest, 192]).T), axis=0).T    
    pred_pad = (pred_pad1 + pred_pad2*1j) * E_lim/4
    pred_pad = pred_pad[:, 0:Npred]

    IFFT = np.fft.irfft(pred_pad)
    
    Ex = E_x[0:E_lim, 0]

    fname= save_directory + "/FEFF_predictions.mat"
    scipy.io.savemat(fname, {'XANES_x':X_x[0, :],
                             'XANES_y':U_test_f,
                             'FFT_x':X_test_f,
                             'FFT_pred':S_pred_f,
                             'FFT_test':FFT_test_f,
                             'IFFT_y':S_test_f,
                             'IFFT_y_pred':IFFT,
                             'IFFT_x':Ex})





