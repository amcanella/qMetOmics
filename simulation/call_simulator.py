# -*- coding: utf-8 -*-
"""
Created on May 29 2024

@author: Alonso

Coordinates data loading and spectrum generation, and handles progress and
 optional saving of outputs.
"""
import time
import simulator
from simulator import ProcessData, Simulator

p = ProcessData('C:/path/to/database.xlsx') #INPUT YOUR DATABASE PATH, template of database will be provided 
d = p.create_dict()
s = Simulator(dictionary = d, met_data=p.met_data, clust_data = p.clust_data, clust_dict=p.clust_dict())

print(f'this value {p.met_data[0][6]} is always 0.0007')

NFREQ = 22_473 #ZEROS REMOVED FROM 32_768 POINTS 
SAMPLES = 10_000 #Normally batches of 10k samples that weigh around 4 GB
#NO UREA
mets = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,61,62,63,65,66,67]
NMET = len(mets)
NOISE = 0.0001
SPUR_FLAG = 1  # 0 = no spur; 1 = include spurious peaks to add noise

#Number the spectra batches
for j in range(10,11):

    print(j)
    spectrum = [0]*SAMPLES
    conc = [0]*SAMPLES
    aligned = [0]*SAMPLES
    spur = [0]*SAMPLES

    start = time.perf_counter()
    for i in range(SAMPLES):

        if i in (SAMPLES/2, SAMPLES/4):
            print(f'\n {i} Almost there!')

        spectrum[i], conc[i], aligned[i], spur[i] = s.constructor(mets, noise = NOISE, spur_flag = SPUR_FLAG)  #, aligned[i]

    end =  time.perf_counter()
    timer = (end - start )/60

    print(f'\nDone with simulating! Time: {timer}')

    # s.csv_gen( f'spectra/x_{SAMPLES}_{NMET}_{NFREQ}_{j}_GLYBIEN.csv', NFREQ, spectrum)
    # s.csv_gen( f'spectra/y_met_{SAMPLES}_{NMET}_{NFREQ}_{j}_GLYBIEN.csv', 67, conc) #LEAVE AS 67
    # s.csv_gen( f'spectra/y_aligned_{SAMPLES}_{NMET}_{NFREQ}_{j}_GLYBIEN.csv', NFREQ, aligned)
    print('----------------------------------------------------')
