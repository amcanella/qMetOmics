# -*- coding: utf-8 -*-
"""
Created on March 20 2024

@author: Alonso

Generates synthetic 1H NMR spectra from a metabolite/cluster database.

"""
import math
import random
import csv
import pandas as pd
import numpy as np


class ProcessData:

    def __init__(self, file_name):
        """Initializes the simulator."""
        self.file_name = file_name

        self.met_df = pd.read_excel(self.file_name, 'Mets')
        clust_df = pd.read_excel(self.file_name, 'Clusters')
        peak_df = pd.read_excel(self.file_name, 'Peaks')

        self.met_data = self.met_df.values
        self.clust_data = clust_df.values
        self.peak_data = peak_df.values


    def list_maker(self, array):
        """Constructs peaks list."""
        list_peaks = [list(a) for a in array]
        for row in list_peaks:
            #Insert name of the met
            row.insert(1, self.met_data[int(row[0])-1][1])
            #Remove amplitude
            row.remove(row[5])

        return  list_peaks

    #Create a dictionary with the peaks info, more accessible
    def create_dict(self):
        """Creates a nested dictionary of peaks grouped by metabolite and cluster."""
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

    def clust_dict(self):
        """Creates a cluster dictionary"""
        clust_d = {}

        for row in self.clust_data:

          key = row[0]

          if key not in clust_d:
            clust_d[row[0]] = [list(row)]
          else:
            clust_d[row[0]].append(list(row))

        return clust_d


class Simulator:
    """Builds raw and aligned spectra by summing Lorentzian peaks for selected metabolites.
    Generates per-sample random cluster shifts within allowed ppm ranges and adds optional noise/spurious peaks."""
    def __init__(self, *, met_data, clust_data, clust_dict, dictionary = None):

        self.dictionary = {} if dictionary is None else dictionary
        self.met_data = met_data
        self.clust_data = clust_data
        self.clust_dict = clust_dict


    #Vary the width
    def set_width(self, width):
        """"Applies a realistic randomised peak width change within ranges."""
        spectral_f = 500 #samples are taken at 500 MHz

        width_norm = width/spectral_f #scaled width by the reference 500 MHz

        return width_norm*random.gauss(1, 0.04) #4% small variation to the width of the peak

    def set_new_centre(self, clusters):
        """Applies a random shift for each cluster within its allowed ppm window."""
        id_ = 0
        lista = []
        for row in clusters:
            met_id = row[0]

            #cluster ranges
            rango0=row[2]  #before it used to be 0
            rango1=row[3]  #before it used to be 0.04
            sigma = np.abs((rango1 - rango0)/2) #0.02
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
        """Computes a Lorentzian line shape."""
        return ((conc/conc_ref)*(2*gamma*area_n)/(np.pi*((gamma**2)+4*(x-x0)**2)))

    #Make the zero areas
    def ranges(self, a):
        """Cuts the tails of the spectra."""
        b = a[5557:28030] #Cutting tails with zero values

        return b

    def csv_gen(self, csv_name, points, matrix):
         """Writes a matrix-like object to CSV."""
         titles =[f'V{i}' for i in range(0, points)]
         with open(csv_name, 'w', newline='') as csvfile:
             csv_writer=csv.writer(csvfile)
             csv_writer.writerow(titles)
             csv_writer.writerows(map(np.ndarray.tolist, matrix)) if isinstance(matrix[0], np.ndarray) else  csv_writer.writerows(matrix)#applying map function to iterable

    def spur_gen(self,start, end, points):
        """Generates a spurious-peak signal by sampling random clusters and translating peaks."""
        peaks = 0
        lor = 0
        x_new = np.linspace(start, end, points) #np.linspace(0.3807, 9.9946, 22473)

        while peaks < 21:

          sample_met = random.sample(list(range(1, 64)) + list(range(65, 68)), 1)[0]

          while True:
            centre = random.gauss(5, 5)
            if 0.3807 <= centre <= 9.9946: #add limits within cut so more challenging for the model
              x_rd = centre
              break
          conc = random.uniform(0, self.met_data[sample_met-1][6]) #max urine concentration
          conc_ref = self.met_data[sample_met-1][5] #concentration reference

          cluster = random.sample(range(1, len(self.dictionary[sample_met])+1), 1)[0] #pick one cluster
          peaks += len(self.dictionary[sample_met][cluster])

          for value in self.dictionary[sample_met][cluster]:

            centre_clust = self.clust_dict[sample_met][cluster-1][5]
            diff = centre_clust - x_rd
            x0 = value[4] - diff #centreSPUR(value[4], x_rd) #value[4]

            gamma = self.set_width(value[5])
            area = value[6]
            lor += self.lorentzian(x_new, x0, gamma, area, conc, conc_ref)

        return lor

    def constructor(self, mets, noise, spur_flag = 0):
            """Constructs synthetic spectrum (raw), with optional noise and spurs, and its corresponding ground truth (concentration + aligned)."""
            d = self.dictionary
            shifts = self.set_new_centre(self.clust_data)
            start = -1.997
            end = 12.024
            points = 32_768
            x = np.linspace(start, end, points)
            raw_spect = 0
            conc_solution_row = [0]*len(self.met_data) #COULD BE LEN mets but leave as it is

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
                        alig_spect += self.lorentzian(x,centre,gamma,area,conc, conc_ref)
            #Add noise
            noise = np.random.normal(0, noise, len(raw_spect))
            #Add spurious peaks
            spur = self.spur_gen(start, end, points) if spur_flag == 1 else np.zeros(points)

            spect_noise = raw_spect + noise + spur
            a_spect_noise = alig_spect + noise + spur

            #Add the zero areas or cut
            spect_cut = self.ranges(spect_noise)
            a_spect_cut = self.ranges(a_spect_noise) #aligned spectrum

            #Normalize to 1
            new_x = np.linspace(0.3807, 9.9946, 22_473)
            integral = np.trapz(spect_cut, new_x)
            spect = spect_cut/integral

            #ALIGNED
            a_integral = np.trapz(a_spect_cut, new_x)
            a_spect = a_spect_cut/a_integral

            return spect, conc_solution_row, a_spect, spur
