
import pandas as pd # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from simulator import process_data
from simulator import Simulator

p = process_data('/imppc/labs/jadlab/amoran/tbOmicsRepo/Metabo_tables_14.xlsx')
d = p.create_dict()

s = Simulator(dictionary = d, met_data=p.met_data, clust_data = p.clust_data)
print(f'\nthis is a test in functions: {p.met_data[0][6]}')

def importSignals(value):

  dfx = pd.read_csv(f'/imppc/labs/jadlab/amoran/Metabolomics/Spectra_generator/spur/x_10000_66_{nfreq}_{value}.csv')

  dfy = pd.read_csv(f'/imppc/labs/jadlab/amoran/Metabolomics/Spectra_generator/spur/y_met_10000_66_{nfreq}_{value}.csv')

  #x = dfx.values
  #y = dfy.values

  return dfx,dfy

nfreq = 22473 #32_768 #52_234
nmet = 66
ndata = 10_000


#TODO IMPORT A TXT
def normalize_mets(y):
  rows = len(y)
  yp = np.zeros([rows, nmet,1])
  c=0 #track vector positionz

  for i in range(len(y[0])):
    if y[0][i]!= 0:
      yp[:,c] = (1/p.met_data[i][6])*y[:,i]
      '''print(1/p.met_data[i][6])
      print(c)
      print(i)'''

      c+=1
    else:
      pass


  '''yp[:, 0] =  1430 * y[:, 0]
  yp[:, 1] = 100 * y[:, 2]
  yp[:, 2] = 2000 * y[:, 3]
  yp[:, 3] =  40 * y[:, 4]
  yp[:, 4] = 115 * y[:, 6]
  yp[:, 5] =  625 * y[:, 7]
  yp[:, 6] =  3 * y[:, 8]
  yp[:, 7] =  525 * y[:, 21]'''

  s = np.sum(yp, axis = 1).reshape((rows, 1,1))
  yp = yp / s

  return yp


import scipy # type: ignore



def customGenerator(lower, upper):
  c = 0
  for j in range(lower, upper+1):

    print(f'\nLoading datasets...{j}')
    dfx, dfy = importSignals(j)

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

def plotLossFunc(current_datetime, history,run_name):
  
  plt.figure()
  plt.xlabel('Epocs')
  plt.ylabel('Amount loss')
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss)+1)
  plt.plot(epochs, loss, 'y', label = 'Training loss')
  plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
  title = 'Training and validation loss'
  plt.title(f'{title}')
  plt.legend()
  #plt.ylim(0,0.000725)
  #plt.xlim(0,54)
  plot_name = f'kt_{current_datetime}_{title}_{run_name}'
  plt.savefig(f'/imppc/labs/jadlab/amoran/tbOmicsRepo/figs/{plot_name}.png')
  plt.show()

def plotGraphs(current_datetime, y_t, y_test, pred_train, pred_test,run_name):

  plt.figure()
  plt.scatter(y_t[:,13], pred_train[:,13], c = 'r', label= 'Prediction training', alpha = 0.5)
  plt.scatter(y_test[:,13], pred_test[:,13], c = 'b', label = 'Prediction test', alpha = 0.5)
  plt.xlabel('real values')
  plt.ylabel('predictions')
  plt.legend()
  title = f'glycine r = {scipy.stats.pearsonr(y_test[:,13], pred_test[:,13])[0]*100:.3f}%'
  plt.title(title)
  plot_name = f'kt_{current_datetime}_{title}_{run_name}'
  plt.savefig(f'/imppc/labs/jadlab/amoran/tbOmicsRepo/figs/{plot_name}.png')
  plt.show()
  
def plotGraphTraining(current_datetime, y_t, pred_train,run_name):

  plt.figure()
  plt.scatter(y_t[:,13], pred_train[:,13], c = 'r', label= 'Prediction training', alpha = 0.5)
  plt.xlabel('real values')
  plt.ylabel('predictions')
  plt.legend()
  y_t = y_t.reshape(len(y_t), nmet)
  title = f'glycine training r = {scipy.stats.pearsonr(y_t[:,13], pred_train[:,13])[0]*100:.3f}%'
  plt.title(title)
  plot_name = f'kt_{current_datetime}_{title}_{run_name}'
  plt.savefig(f'/imppc/labs/jadlab/amoran/tbOmicsRepo/figs/{plot_name}.png')
  plt.show()


def plotScatters(run_number, current_datetime,y_t, y_test, pred_train, pred_test,run_name):
  plt.figure()
  titles = ['hydroxy','acetate','alanine', 'aspartate', 'betaine', 'butyrate', 'creatine', 'creatinine', 'glucose', 'glutamate' ]
  fig, ax = plt.subplots(3,4, figsize = (40,10))
  c=0
  for row in range(len(ax)):
    for i in range(len(ax[0])):
        ax[row, i].scatter(y_t[:,c], pred_train[:,c], c = 'r', label= 'Prediction training', alpha = 0.5) # type: ignore
        ax[row, i].scatter(y_test[:,c], pred_test[:,c], c = 'b', label = 'Prediction test', alpha = 0.5) # type: ignore
        ax[row, i].set_xlabel('real values')
        ax[row, i].set_ylabel('predictions')
        ax[row, i].legend()
        ax[row, i].set_title(f'{d[c+1][1][0][1]} r = {scipy.stats.pearsonr(y_test[:,c], pred_test[:,c])[0]*100:.3f}%') # type: ignore
        c+=1
  plot_name = f'kt_{current_datetime}_scatters_{run_number}_{run_name}'
  plt.savefig(f'/imppc/labs/jadlab/amoran/tbOmicsRepo/figs/{plot_name}.png')
  plt.show()
  
  



