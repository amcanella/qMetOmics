# -*- coding: utf-8 -*-
"""
Created on March 20

@author: Alonso
"""
import pandas as pd
import math
import numpy as np 
import random 
import csv
import matplotlib.pyplot as plt 

class process_data:
    
    def __init__(self, file_name):
        
        self.file_name = file_name
        
        self.met_df = pd.read_excel(self.file_name, 'Mets')
        clust_df = pd.read_excel(self.file_name, 'Clusters')
        peak_df = pd.read_excel(self.file_name, 'Peaks')
        
        self.met_data = self.met_df.values
        self.clust_data = clust_df.values
        self.peak_data = peak_df.values
        

    def list_maker(self, array):
        #Add and remove data from list
        list_peaks = [list(a) for a in array]
        for row in list_peaks: 
            #Insert name of the met
            row.insert(1, self.met_data[int(row[0])-1][1])
            #Remove amplitude 
            row.remove(row[5])   
            
        return  list_peaks        
    
    #Create a dictionary with the peaks info, more accessible       
    def create_dict(self):
        
        list_peaks = self.list_maker(self.peak_data)
         
        peak_dict = {}
        for row in list_peaks:
            key = row[0]
            
            if math.isnan(row[2]):
                pass
            else:
                key_2 = int(row[2])
            if key not in peak_dict:
                inner_dict = {}
                peak_dict[key]=inner_dict
            if key_2 not in inner_dict:
                peak_dict[key][key_2]=[row] #If the key does not exist, create new dict
            else:
                peak_dict[key][key_2].append(row) 
        return peak_dict
    


class Simulator:
    
    def __init__(self, *, dictionary = {}, met_data, clust_data):
        
        self.dictionary = dictionary
        self.met_data = met_data
        self.clust_data = clust_data
        
        
    #Vary the width
    def set_width(self, width):
        spectral_f = 500 #samples are taken at 500 MHz

        width_norm = width/spectral_f #scaled width by the reference 500 MHz  
        return width_norm*random.gauss(1, 0.04) #4% small variation to the width of the peak
    
    #Shift the centres
    def set_new_centre(self, clusters):
        
        id_ = 0
        lista = []
        for row in clusters:
            met_id = row[0]
            
            #cluster ranges
            rango0=row[2]  #before it used to be 0
            rango1=row[3]  #before it used to be 0.04
            print(rango0, rango1)
            sigma = np.abs((rango1 - rango0)/2) #0.02 
            print(sigma)
            clust_centre = row[5]
            
            new_centre = random.gauss(clust_centre, sigma)
            shift = new_centre - clust_centre
            
            
            if met_id == id_ :
                pass
                
            else:
                id_ +=1
                lista.append([id_])
                
            lista[id_ - 1].append(shift)   
            
        return lista
                
         
        
    def lorentzian(self, x,x0,gamma,area_n,conc, conc_ref):
        return ((conc/conc_ref)*(2*gamma*area_n)/(np.pi*((gamma**2)+4*(x-x0)**2))) 
    
    #Make the zero areas
    def ranges(self, a):
    
        a[: 5557] = 0
        a[16107 : 16692] = 0 #Dividir dos espectros?
        a[ 28031 :] = 0
        b = a[5557:28030] #Cutting zeros tails 
        # a[24293:32652] = 0
        return b
    
    def csv_gen(self, csv_name, points, matrix):
         
         titles =[f'V{i}' for i in range(0, points)]
         with open(csv_name, 'w', newline='') as csvfile:
             csv_writer=csv.writer(csvfile)
             csv_writer.writerow(titles)
             csv_writer.writerows(map(np.ndarray.tolist, matrix)) if isinstance(matrix[0], np.ndarray) else  csv_writer.writerows(matrix)#applying map function to iterable

   
    def constructor(self, mets, noise):
            #Add the shift and the width variations and plot
            d = self.dictionary
            shifts = self.set_new_centre(self.clust_data)
            x = np.linspace(-1.997, 12.024, 32_768)
            # x = np.linspace(0.04, 10, 52_234)
            raw_spect = 0
            conc_solution_row = [0]*len(self.met_data)
            
            alig_spect = 0
            
            for m in mets:
                concentration_urine =  self.met_data[m-1][6]
                wished = random.uniform(0, concentration_urine)

                conc_solution_row[m -1] = wished
                
                con_reference = self.met_data[m-1][5]
                
                for key, value in d[m].items():

                    for row in value:
                        
                        centre = row[4]
                        shift = shifts[m-1][key]
                        new_centre = row[4] + shift


                        #Change width to ppm and add variation
                        width_var= self.set_width(row[5])

                        x0= new_centre
                        gamma = width_var
                        area = row[6]
                        conc = wished #wished concentration
                        conc_ref = con_reference
                        #call the lorentzian
                        raw_spect += self.lorentzian(x,x0,gamma,area,conc, conc_ref)
                        
                        #ALIGNED
                        #alig_spect += self.lorentzian(x,centre,gamma,area,conc, conc_ref)
            #Add noise           
            noise = np.random.normal(0, noise, len(raw_spect))
            spect_noise = raw_spect + noise
            
            #a_spect_noise = alig_spect + noise
                        
            #Add the zero areas or cut
            spect_cut = self.ranges(spect_noise)
            
            # a_spect_cut = self.ranges(a_spect_noise)
            
            #Normalize to 1
            new_x = np.linspace(0.3807, 9.9946, 22_473)
            integral = np.trapz(spect_cut, new_x)
            spect = spect_cut/integral
            
            # plt.plot(new_x, spect)
            # plt.xlim(10, 0)
            # plt.show()
            # a_integral = np.trapz(a_spect_cut, new_x)
            # a_spect = a_spect_cut/a_integral
            
            return spect, conc_solution_row #, a_spect
    