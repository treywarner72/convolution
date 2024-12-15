import numpy as np
from numba import njit
from math import sqrt, pi, exp
from .fitting import Fit_function

@njit
def normal_pdf(x, mean, sigma):
    # x: array, mean and sigma: scalars
    out = np.empty_like(x)
    factor = 1.0 / (sigma * sqrt(2.0 * pi))
    for i in range(x.size):
        z = (x[i] - mean) / sigma
        out[i] = factor * exp(-0.5 * z * z)
    return out

@njit
def trapz(y, x):
    s = 0.0
    for i in range(x.size - 1):
        dx = x[i+1] - x[i]
        s += 0.5 * (y[i] + y[i+1]) * dx
    return s

@njit
def compute_dist(val, c, d_x, max_mother, mu, sigma):
    dist = np.zeros_like(val)
    for i in range(d_x.size):
        dist += c[i] * normal_pdf(val + max_mother - mu, d_x[i], sigma)
    return dist

class Convolution:
    def __init__(self, mother, max_mother, bins):
        self.mother = mother
        self.max_mother = max_mother
        self.c, self.bins = np.histogram(mother, bins)
        self.d_x = 0.5 * (self.bins[1:] + self.bins[:-1])

    def pdf(self, n=(1, np.inf), mu=(-np.inf, np.inf), sigma=(0, 1, np.inf)):
        return Fit_function(self._pdf, [n, mu, sigma])

    def _pdf(self, val, n, mu, sigma):
        val = np.atleast_1d(val).astype(np.float64)
        dist = compute_dist(val, self.c.astype(np.float64), self.d_x.astype(np.float64), float(self.max_mother), float(mu), float(sigma))
        integral = trapz(dist, val)
        return (n / integral) * dist

    @staticmethod
    def tanh_bin(bin_c, bin_min, bin_max, scale, assymtote):
        b = (bin_max - bin_min - assymtote * bin_c) / np.tanh(bin_c / scale)
        return bin_min + b * np.tanh(np.arange(0, bin_c+1)/scale) + assymtote * np.arange(0, bin_c+1)
