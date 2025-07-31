# qMetOmics



Spectra\_generator: python-based tool for 1H NMR spectra simulation based on official databases. 



## Spectra_generator

\- User must input .xlsx or .csv files containing metabolite names, peak parameters, and concentrations.



\- Generates realistic low- and high-field NMR spectra for each sample.



\- Optimized for urine samples, taking into account pH-dependent chemical shift variations.


## NN

 - Takes the simulations and trains NN
 - Quant model outputs a 66-dimension vector with relative metabolite concentrations
 - Alignment model outputs the same spectrum with clusters in their standard position
 - Plots performance parameters
