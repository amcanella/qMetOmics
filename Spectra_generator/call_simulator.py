# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:27:28 2024

@author: Alonso
"""
import simulator
from simulator import process_data 
from simulator import Simulator
import time 

p = process_data('C:/txts/Metabo_tables_14.xlsx') #INPUT YOUR DATABASE PATH, template of database will be provided in the future
d = p.create_dict()
s = Simulator(dictionary = d, met_data=p.met_data, clust_data = p.clust_data, clust_dict=p.clust_dict())

print(f'this is a test {p.met_data[0][6]}')

NFREQ = 22_473 #ZEROS REMOVED 32_768 #52_234 #32768
SAMPLES = 1 #Normally batches of 10k samples that weigh around 4 GB
#NO UREA
mets = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,61,62,63,65,66,67] #np.arange(1,26) #[3,4,5]
# mets = [3,4,5]
nmet = len(mets)
noise = 0.0001
spur_flag = 1  # 0 = no spur; 1 = include spurious peaks to add noise

for j in range(10,11):
    
    print(j)
    spectrum = [0]*SAMPLES
    conc = [0]*SAMPLES
    aligned = [0]*SAMPLES
    spur = [0]*SAMPLES
    
    start = time.perf_counter()
    for i in range(SAMPLES):
    
        if i == SAMPLES/2 or i==SAMPLES/4 :
            print(f'\n {i} We are half way through!')
          
        spectrum[i], conc[i], aligned[i], spur[i] = s.constructor(mets, noise, spur_flag = spur_flag)  #, aligned[i]
        
    end =  time.perf_counter()
    timer = (end - start )/60
    
    print(f'\nDone with simulating! Time: {timer}')
    
    # s.csv_gen( f'aligned/x_{SAMPLES}_{nmet}_{NFREQ}_{j}_GLYBIEN.csv', NFREQ, spectrum)
    # s.csv_gen( f'aligned/y_met_{SAMPLES}_{nmet}_{NFREQ}_{j}_GLYBIEN.csv', 67, conc) #leave as 67
    
    # s.csv_gen( f'aligned/y_aligned_{SAMPLES}_{nmet}_{NFREQ}_{j}_GLYBIEN.csv', NFREQ, aligned)
    print('----------------------------------------------------')
    



# import matplotlib.pyplot as plt 
# import numpy as np 

# x = np.linspace(0.3807, 9.9946, 22_473)
# plt.plot(x, spectrum[0], label = 'shifted')
# plt.plot(x, aligned[0], label = 'aligned')
# plt.legend(loc= 'upper right')
# plt.show()