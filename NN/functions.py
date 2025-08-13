import pandas as pd
import numpy as np
from simulator import process_data, Simulator

NFREQ = 22473 #32_768
NMET = 66

p = process_data('path/to/database.xlsx')
d = p.create_dict()
s = Simulator(dictionary = d, met_data=p.met_data, clust_data = p.clust_data)

def import_signals(value):
  """Reads one pair of files (X,Y) for the given index and return DataFrames or arrays."""
  dfx = pd.read_csv(f'samples/x_10000_66_{NFREQ}_{value}.csv')

  dfy = pd.read_csv(f'samples/y_met_10000_66_{NFREQ}_{value}.csv')

  return dfx,dfy

#TODO IMPORT A TXT
def normalize_mets(y):
  """Normalizes each metabolite column by its reference factor from p.met_data[:,6]."""
  rows = len(y)
  yp = np.zeros([rows, NMET,1])
  c=0 #track vector positions

  for i in range(len(y[0])):
    if y[0][i]!= 0:
      yp[:,c] = (1/p.met_data[i][6])*y[:,i]
      c+=1
    else:
      pass

  s = np.sum(yp, axis = 1).reshape((rows, 1,1))
  yp = yp / s

  return yp

def custom_generator(lower, upper):
  """Loads and concatenates multiple indices into single X and Y arrays."""
  c = 0
  for j in range(lower, upper+1):

    print(f'\nLoading datasets...{j}')
    dfx, dfy = import_signals(j)

    if c == 0:
      dfx_total = dfx
      dfy_total = dfy
      c+=1
    else:
      dfx_total = pd.concat([dfx_total, dfx], axis=0)
      dfy_total = pd.concat([dfy_total, dfy], axis=0)

  x = dfx_total.values
  y = dfy_total.values

  return x, y
