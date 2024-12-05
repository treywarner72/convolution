import scipy
import numpy as np

class Convolution:
    def __init__(self,mother, max_mother, bins):
        self.mother = mother
        self.max_mother = max_mother
        self.c, self.bins= np.histogram(mother, bins)
        self.d_x = 0.5*(self.bins[1:] + self.bins[:-1])

    def pdf(self, n=(1,np.inf), mu=(-np.inf,np.inf), sigma=(0,1,np.inf)):
        return Fit_function(self._pdf, [n,mu,sigma])

    def _pdf(self,val,n,mu,sigma):
        dist = sum([self.c[i] * scipy.stats.norm.pdf(val + self.max_mother-mu, self.d_x[i],sigma) for i in range(len(self.d_x))])
        return n / np.trapz(dist, val) * dist

def tanh_bin(bin_c,bin_min,bin_max, scale,assymtote):
    b = (bin_max-bin_min-assymtote*bin_c)/np.tanh(bin_c/scale)
    return bin_min + b*np.tanh(np.arange(0,bin_c+1)/scale) + assymtote * np.arange(0,bin_c+1)