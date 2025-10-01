
# README for The Convolutional Method for Fitting Long-Tailed Distributions

This repository contains the code required to reproduce the results presented in the paper:

> Trey Warner, Thomas Boettcher, and Michael Sokoloff. *"The Convolutional Method for Fitting Long-Tailed Distributions"*

---

## Fitle

[fitle](https://github.com/trey-warner/fitle) is my Python library for building and fitting statistical models in a symbolic way.  

The convolutional method described in this paper is implemented using fitle.

## Data

In the data directory you will find 
* MERR.root - an emperical error distribution used for monte carlo
* mc/ - data for Monte Carlo 
	* raw/ includes the raw ROOT files for the two peaks (Dp, Ds) for D->KKp and the one for D+ -> Kpp
	* histogram/ includes .npy files of these binned into the histogram described in the paper
	* subsets/ was used in the past to avoid having to upload the raw ROOT files to github, but it is currently unused
* observed/ - contains .npy files for the finely cut observed distributions from early run 3 data.

mc_results/ contains text files with the results of the Monte Carlo.

## Notebooks

In the notebooks directory you will find
* fitle.ipynb - run this to reinstall fitle, which is used for the fits
* binning_scheme.ipynb - generates the plots regarding binning in the paper, and also generates the histograms of the PHOTOS simulations
* real_data_kkp.ipynb - all graphs and tables regarding real KKp data are generated here
* real_data_kpp.ipynb - all graphs and tables regarding real Kpp data are generated here
* monte_carlo.ipynb - shows the Monte Carlo process and some unused graphs. The full Monte Carlo is done in monte_carlo.py
* monte_carlo_analysis.ipynb - generates the plots regarding Monte Carlo results and has many unused charts.
