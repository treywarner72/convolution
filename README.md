# README for The Convolutional Method for Fitting Long-Tailed Distributions

This repository contains the code required to reproduce the results presented in the paper:

> Trey Warner, Michael Sokoloff, and Thomas Boettcher. *"The Convolutional Method for Fitting Long-Tailed Distributions"*, December 2024.

The library implements convolutional methods to model long-tailed distributions resulting from processes like radiative tails and precisely determine the peak position.

---

## Fitlib

The fits in the paper are performed with a custom fitting library `fitlib`. Documentation is provided in the source code and the demonstration `fitlib_demo.ipynb.`

## Reproducing Results

Monte carlo data may be generated and fit to reproduce Figures 7 and 8 in the paper with `monte_carlo.ipynb`.
