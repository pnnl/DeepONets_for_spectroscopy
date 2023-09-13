#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2023
@author: Amanda Howard (amanda.howard@pnnl.gov)

USAGE: 
    For plotting the results of training using the DeepONet code to predict
    FEFF profiles

"""

import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
import scipy.io

if __name__ == "__main__":

    #Location where data is stored
    test_str = "../Trained_models/Trained_0_6/"
    #Index to plot
    beta = 0


    with open(test_str + 'input.txt') as f:
        lines = f.readlines()

    xstart = int(lines[0].strip())
    xend =  int(lines[1].strip())

    test_set = [0, 5, 11]
    train_set = [1,2, 3, 4, 6, 7, 8, 9, 10, 12]
    
    net_data_dir = test_str + "/"
    d_vx = scipy.io.loadmat(net_data_dir+"FEFF_predictions.mat")
    XANES_y, XANES_x, EXAFS_x, EXAFS_y,  EXAFS_pred, \
        FFT_x, FFT_y, FFT_pred = (d_vx["XANES_y"].astype(np.float32), 
               d_vx["XANES_x"].astype(np.float32), 
                d_vx["IFFT_x"].astype(np.float32),
                d_vx["IFFT_y"].astype(np.float32),
                d_vx["IFFT_y_pred"].astype(np.float32),
                d_vx["FFT_x"].astype(np.float32),
                d_vx["FFT_pred"].astype(np.float32),
                d_vx["FFT_test"].astype(np.float32))
    
    XANES_x = XANES_x[0, xstart:xend]
    FFT_x = FFT_x[0, :, 0]
    EXAFS_x = EXAFS_x[0, :]
    NFFT = FFT_x.shape[0]
    lrange = np.arange(40)
    Nplot=EXAFS_x.shape[0]
    xrange = 14

    mag_s = np.sqrt(FFT_y[:, :, 0]**2 + FFT_y[:, :, 1]**2)
    mag_pred = np.sqrt(FFT_pred[:, :, 0]**2 + FFT_pred[:, :, 1]**2)
    

        
    
    #Plot figure
    fig3, gx = plt.subplots(figsize=(12, 3))
    plt.figure(fig3.number)
    plt.subplot(1,3,1)
    plt.plot(XANES_x, XANES_y[beta, 0, :]+1, 'k', linestyle='-', label='Data')
    plt.plot([-1, 0], [0, 0], '#4e79a7', linestyle='--', label='Prediction')
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.xlim([8979.093, 9114.096])
    plt.legend(fontsize=12)#, bbox_to_anchor=(1.05, 1))
    plt.title('XANES', fontsize=14)

 
    plt.subplot(1,3,2)
    plt.plot(FFT_x[lrange], mag_s[beta, lrange], 'k', linestyle='-', label='data')
    plt.plot(FFT_x[lrange], mag_pred[beta, :], '#4e79a7', linestyle='--', label='prediction')
    plt.tick_params(labelsize=16)
    plt.title('Scaled FFT', fontsize=14)
     
    plt.subplot(1,3, 3)
    plt.plot(EXAFS_x[0:Nplot], EXAFS_y[beta, 0:Nplot, 0], 'k', linestyle='-', label='data')
    plt.plot(EXAFS_x[0:Nplot], EXAFS_pred[beta,0:Nplot], '#4e79a7', linestyle='--', label='prediction')
    plt.xlim([0, xrange])
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.title('EXAFS', fontsize=14)

    plt.savefig(net_data_dir+'/Pred_' + str(beta) + '.png', format='png', dpi=200)
        


     #If True, plot losses
    loss = True
    if loss:
        d_vx = scipy.io.loadmat(net_data_dir +'losses.mat')
        train,test = ( d_vx["training_loss"].astype(np.float32),
                 d_vx["testing_loss"].astype(np.float32))
        
        fig1, ax = plt.subplots()
        plt.figure(fig1.number)
        step = np.arange(0, 1000*len(train[0]), 1000)
        plt.semilogy(step, train[0], '#59a14f', linestyle='-', label='Loss')
        plt.semilogy(step, test[0], 'k', linestyle='-', label='Test')
        plt.xlabel('Number of Epochs', fontsize=20)
        plt.ylabel(r'Loss', fontsize=20)
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1))
        plt.tick_params(labelsize=16)
        plt.tight_layout()
       # plt.ylim([.01, 10])
        plt.savefig(net_data_dir+'Loss.png', format='png')
        


