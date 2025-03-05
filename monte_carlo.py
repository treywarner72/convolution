import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import fitlib
import vector

f = uproot.open("./data/MERR.root")

Dp_DTF_MERR = f['DecayTree']["Dp_DTF_MERR"].array()

Dp_mother_mass = np.fromfile('data/Dp_mother_mass')
Ds_mother_mass = np.fromfile('data/Ds_mother_mass')


def sample():
    raw_data_dp = np.random.choice(Dp_mother_mass, size=800000) + np.multiply(np.random.choice(Dp_DTF_MERR,800000),np.random.randn(800000)) 
    raw_data_ds = np.random.choice(Ds_mother_mass, size=1000000) + np.multiply(np.random.choice(Dp_DTF_MERR,1000000),np.random.randn(1000000))
    background = np.random.exponential(150, 60000)+1840
    return np.concatenate([raw_data_dp,raw_data_ds,background])


tanh_dp = fitlib.Convolution([0], 1869.65, 1)
Dp_data = np.loadtxt("./data/Dp_histogram.csv", delimiter=",", skiprows=1)
tanh_dp.d_x = Dp_data[:, 0]
tanh_dp.c = Dp_data[:, 1]

tanh_ds = fitlib.Convolution([0], 1968.33, 1)
Ds_data = np.loadtxt("./data/Ds_histogram.csv", delimiter=",", skiprows=1)
tanh_ds.d_x = Ds_data[:, 0]
tanh_ds.c = Ds_data[:, 1]


import numba

from fitlib import f


fitters_conv = []


for i in range(3000):
    print('convo')
    montecarlo = sample()
    fitter = fitlib.Fitter.binned(montecarlo, bins=200,range=(1840,2040))
    fitter.mu = f(1950,2000)
    fitter.diff = f(0,100,200)

    fitter.pdf = [
        tanh_dp.pdf(mu=fitter.mu - fitter.diff),tanh_dp.pdf(mu=fitter.mu - fitter.diff),
        tanh_ds.pdf(mu=fitter.mu),tanh_ds.pdf(mu=fitter.mu),
        fitlib.Exp(x0=1840)
    ]

    if fitter.chi2(1000000).valid:
        fitters_conv.append(fitter)
        np.array([fit.mu.value for fit in fitters_conv]).tofile('mc_results/conv_mu_val')
        np.array([fit.mu.error for fit in fitters_conv]).tofile('mc_results/conv_mu_err')
        np.array([fit.diff.value for fit in fitters_conv]).tofile('mc_results/conv_diff_val')
        np.array([fit.diff.error for fit in fitters_conv]).tofile('mc_results/conv_diff_err')
        print('successo!')





