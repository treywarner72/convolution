import numpy as np
from numba import njit
from math import sqrt, pi, exp
from .fitting import Fit_function

@njit
def normal_pdf(x, mean, sigma):
    """
    Efficient NJIT compiled normal distribution
    """
    out = np.empty_like(x)
    factor = 1.0 / (sigma * sqrt(2.0 * pi))
    for i in range(x.size):
        z = (x[i] - mean) / sigma
        out[i] = factor * exp(-0.5 * z * z)
    return out

@njit
def trapz(y, x):
    """
    Efficient NJIT trapezoidal integration
    """
    s = 0.0
    for i in range(x.size - 1):
        dx = x[i+1] - x[i]
        s += 0.5 * (y[i] + y[i+1]) * dx
    return s


@njit
def compute_dist(val, c, d_x, mass_mother, mu, sigma):
    """
    Efficient NJIT computation of the emperical distribution given:
    val : numpy array of input values
    c : numpy array of counts in the mother mass histogram
    d_x : numpy array of bin widths in the mother mass histogram
    mass_mother : true mass of the mother distribution
    mu : center of the generated observed distribution
    sigma : spread of the generated observed distribution
    """
    dist = np.zeros_like(val)
    for i in range(d_x.size):
        dist += c[i] * normal_pdf(val + mass_mother - mu, d_x[i], sigma)
    return dist

class Convolution:
    """
    A class representing a convolution of a simulated distribution with a resolution function

    Attributes
    ----------
    mother : numpy.ndarray
        An array of the mother distribution to be convolved
    mass_mother : float64
        The true mass of the mother distribution, known a priori
    bins : numpy.ndarray
        An array of bin edges to bin the mother distribution with


    Methods
    -------
    pdf(n=(1, np.inf), mu=(-np.inf, np.inf), sigma=(0, 1, np.inf)):
        Returns a Fit_function parameratized by the Fit_value objects n, mu, and sigma. To be used by a Fitter

    """

    def __init__(self, mother, mass_mother, bins):
        self.mother = mother
        self.mass_mother = mass_mother
        self.c, self.bins = np.histogram(mother, bins)
        self.d_x = 0.5 * (self.bins[1:] + self.bins[:-1])

    def pdf(self, n=(1, np.inf), mu=(-np.inf, np.inf), sigma=(0, 1, np.inf)):
        """
        Returns a Fit_function of the convolution distribution parameratized by the Fit_value objects in order:
        n : cumulative density
        mu : measure of location
        sigma : measure of spread
        """
    
        return Fit_function(self._pdf, [n, mu, sigma])

    # Returns the aggregate pdf value for a given x and parameter values
    def _pdf(self, val, n, mu, sigma):
        val = np.atleast_1d(val).astype(np.float64)
        dist = compute_dist(val, self.c.astype(np.float64), self.d_x.astype(np.float64), float(self.mass_mother), float(mu), float(sigma))
        integral = trapz(dist, val)
        return (n / integral) * dist

    @staticmethod
    def tanh_bin(n_bins, m_min, m_max, alpha, beta):
        """
        Returns a list of bin edges subject to:
        n_bins : number of bins
        m_min : value of the left edge of the left-most bin
        m_max : value of the right edge of the right-most bin
        alpha : the asymptote, the constant that the bin widths approach as n approaches n_bins
        beta : the scale, or how fast the bin widths shrink to approach alpha, where the smaller beta is, the faster the bin widths approach alpha
        """
        A = (m_max - m_min - alpha * bin_c) / np.tanh(n_bins / beta)
        return m_min + A * np.tanh(np.arange(0, n_bins+1)/beta) + alpha * np.arange(0, n_bins+1)
