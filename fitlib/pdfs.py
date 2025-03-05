from .fitting import Fit_function
import numpy as np
import scipy

def Normal(n=(1,np.inf), mu=(-np.inf,np.inf), sigma=(0,1,np.inf)):
    """
    Returns a Fit_function of a Normal distribution with Fit_value objects in order:
    n : cumulative density
    mu : measure of location
    sigma : measure of spread
    """
    return Fit_function(lambda x,n, mu, sigma: n*scipy.stats.norm.pdf(x,mu, sigma), [n,mu,sigma])   

def Exp(n=(0,1,np.inf), x0=(-np.inf,np.inf), a=(0,1,np.inf)):
    """
    Returns a Fit_function of an Exponential distribution with Fit_value objects in order:
    n : cumulative density
    x0 : measure of location
    a : measure of spread
    """
    return Fit_function(lambda x,n,x0,a: n*scipy.stats.expon.pdf(x,x0,a), [n,x0,a])
