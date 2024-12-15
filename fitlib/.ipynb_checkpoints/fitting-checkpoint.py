import numpy as np
from inspect import signature
import iminuit
from iminuit import cost
from .utils import flatten
from tabulate import tabulate
import matplotlib.pyplot as plt

class Fit_param:
    """
    A class representing a single fit parameter with bounds and initial value. Fitted value are filled after fitting.

    Starting Attributes
    -------------------
    min : float
        The minimum allowed value for the parameter (default: -infinity).
    start : float
        The starting value of the parameter (default: 0).
    max : float
        The maximum allowed value for the parameter (default: infinity).

    Filled Attributes
    -----------------
    value : float
        The fitted value of the parameter after optimization.
    error : float
        The error associated with the fitted value.
    """
    def __init__(self,min=-np.inf, start=0, max=np.inf):
        self.min = min
        self.start = start
        self.max = max

    # This method is used internally to treat Fit_param and Fit_params the same way.
    def parametize(self):
        return self

    # The following allow for operations between a Fit_param object and another value to create a Fit_params object with that operation.
    
    def __add__(self,other):
        return Fit_params([self,other], lambda a: a[0]+a[1])
    def __radd__(self,other):
        return Fit_params([self,other], lambda a: a[0]+a[1])

    def __mul__(self,other):
        return Fit_params([self,other], np.prod)
    def __rmul__(self,other):
        return Fit_params([self,other], np.prod)
    
    def __sub__(self,other):
        return Fit_params([self,other], lambda a: a[0]-a[1])
    def __rsub__(self,other):
        return Fit_params([other,self], lambda a: a[0]-a[1])

    # This returns the bounds and starting value for an unfit parameter and the value and error for a fit parameter
    def __str__(self):
        if hasattr(self,'value'):
            return f"{self.value} ± {self.error}" 
        return f"{({self.min}, {self.start}, {self.max})}"

# This function is mostly useful for defining parameters outside of a pdf function, especially when you need to apply an operation to it
def f(min=None, start=None, max=None):
    """
    Returns a Fit_param object.
    f(val) -> constant parameter
    f(min, max) -> bounds. start is the mean unless either bound is infinite
    f(min, start, max) -> bounds and starting value
    """
    # This is used in pdf functions to avoid typing fitlib.f() ad nauseum
    if isinstance(min, tuple):
        min,start,max = (min + (None, None, None))[:3]
    if min is not None and start is not None and max is not None:
        return Fit_param(min, start, max)
    elif min is not None and start is not None:
        if min == -np.inf and start == np.inf:
            return Fit_param(min, 0, start)
        if min == -np.inf:
            return Fit_param(min, start, start)
        if start == np.inf:
            return Fit_param(min, min, start)
        return Fit_param(min, (min+start)/2, start)
    elif min is not None:
        # Again, so that you can put a Fit_param object in the argument of a pdf functio.
        return min if isinstance(min, (Fit_param, Fit_params)) else Fit_param(min, min, min)
    return Fit_param(-np.inf, 0, np.inf)


class Fit_params:
    """
    A collection of Fit_param objects whereon an operation is done before returning a value used by a pdf function.
    
    Attributes
    ----------
    params : float
        A list of Fit_param objects or other objects (eg. floats).
    operation : float
        A function that returns a value given the params array.
    """
    def __init__(self,params,operation):
        self.params=params
        self.operation=operation

    # Returns all the parameters contained in this Fit_params
    def parametize(self):
        return list(filter(None,[p.parametize() if isinstance(p, (Fit_params,Fit_param)) else None for p in self.params]))

    # This is used internally to allow a Fit_params output to be plugged into a pdf function.
    @property
    def val(self):
        return self.operation([p.val if isinstance(p, (Fit_params,Fit_param)) else p for p in self.params] )

    # Returns a list of the fitted values of each contained Fit_param object
    @property
    def value(self):
        return [p.value for p in self.params if isinstance(p, (Fit_params,Fit_param))]

    # Returns a list of the fitted errors of each contained Fit_param object
    @property
    def error(self):
        return [p.error for p in self.params if isinstance(p, (Fit_params,Fit_param))]


    # The following allow for operations between a Fit_params object and another object to create a bigger Fit_params object
    
    def __add__(self,other):
        return Fit_params([self,other], lambda a: a[0]+a[1])
    def __radd__(self,other):
        return Fit_params([self,other], lambda a: a[0]+a[1])

    def __mul__(self,other):
        return Fit_params([self,other], np.prod)
    def __rmul__(self,other):
        return Fit_params([self,other], np.prod)
    
    def __sub__(self,other):
        return Fit_params([self,other], lambda a: a[0]-a[1])
    def __rsub__(self,other):
        return Fit_params([other,self], lambda a: a[0]-a[1])

class Fit_function:
    """
    A class representing a functional form with associated parameters for fitting.

    The `Fit_function` class encapsulates a function (`fcn`) and its parameters,
    allowing for easy evaluation, parameterization, and use in fitting routines.

    Attributes
    ----------
    fcn : callable
        The functional form to be evaluated. It should accept an `x` value and
        one or more parameters as input.
    _params : list of Fit_param
        A list of `Fit_param` instances representing the parameters of the function.

    Methods
    -------
    params():
        Returns a flattened list of all `Fit_param` instances associated with the function.
    call(x):
        Evaluates the function at a given `x` value using the current parameter values. After fitting, these values are the fitted values
    """
    def __init__(self,fcn, params):
        self.fcn=fcn
        self._params=[f(p) for p in params]

    def params(self):
        return flatten([x.parametize() for x in self._params])
        
    def call(self,x):
        return self.fcn(x,*[p.val for p in self._params])


class Fit_values(dict):
    """Container namespace for parameters; .key and ['key'] both work
    """
    def __init__(self, arg):
        super(Fit_values, self).__init__(arg)
        for k, v in arg.items():
            self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Fit_values, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

class Fitter:
    """
    A class for fitting data with a build your own probability density function.

    The `Fitter` class provides utilities for doing Chi^2 minimization fits and unbinned maximum likelihood estimation, in addition visualizing the data and the fit.

    Attributes
    ----------
    arr : numpy.ndarray
        The data array to be fit.
    range : tuple
        The range of the data to consider.
    pdf : list of Fit_function
        The list of probability density functions to fit to the data.

    The following are set with the bin function or when a Fitter is created using Fitter.binned(arr, bins, range)
    ----------
    y : numpy.ndarray
        The counts in each histogram bin.
    x : numpy.ndarray
        The centers of the histogram bins.
    widths : numpy.ndarray
        The widths of the histogram bins.
    err_b : numpy.ndarray
        The statistical errors (sqrt of counts) for each bin.

    Methods
    -------
    chi2(ncall=None):
        Perform a chi-squared fit and update parameter values and errors.
    values():
        Retrieve the fitted values of the parameters.
    errors():
        Retrieve the errors of the fitted parameters.
    plot():
        Visualize the histogram and the fitted curve.
    summary():
        Print a summary of the fit, including parameter values and errors.
    """

       
    def __init__(self, arr, range=None):
        self.range = (np.min(arr),np.max(arr)) if range == None else range
        self.arr = arr[(arr > self.range[0]) & (arr < self.range[1])]

    def bin(self,bins):
        y, edges = np.histogram(self.arr, range=self.range, bins=bins)
        self.y = np.array(y)
        edges = np.array(edges)
        self.x = 0.5*(edges[:-1] + edges[1:])
        self.widths = edges[1:] - edges[:-1]
        self.err_b = np.sqrt(self.y)
        return self

    # An alternate to __init__ that bins a histogram by default.
    @staticmethod
    def binned(arr,bins, range=None):
        ret = Fitter(arr, range)
        return ret.bin(bins)
    

    # pdf is a faux-variable, where fit_params is updated to include the total amount of parameters whenever pdf is updated
    
    @property
    def pdf(self):
        return self.pdf_l
    
    @pdf.setter
    def pdf(self, pdf_l):
        self.pdf_l = pdf_l
        self.fit_params = list(dict.fromkeys(np.concatenate([p.params() for p in self.pdf_l]))) 

    # Returns the aggregate pdf value for a given x and parameter values
    def _pdf(self, x, *args):
        for i in range(len(args)):
            self.fit_params[i].val = args[i]
        
        return sum([fcn.call(x) for fcn in self.pdf_l])

    def _chi2(self, *args):
        "Returns a Chi^2 value given parameter values. This function is minimized with iminuit"
        diff = self.y - self.widths * self._pdf(self.x, *args)
        return (diff[self.err_b > 0]**2 / self.err_b[self.err_b > 0]**2).sum()
    
    def chi2(self,ncall=None):
        start = [p.start for p in self.fit_params]
        limits = [(p.min, p.max) for p in self.fit_params]
        
        self.minimizer = iminuit.Minuit(self._chi2, *start)
        self.minimizer.limits = limits
        ret = self.minimizer.migrad(ncall)

        for i in range(len(self.fit_params)):
            self.fit_params[i].value = self.minimizer.values[i]
            self.fit_params[i].error = self.minimizer.errors[i]

        return ret

    def _r2(self, *args):
        "Returns a Chi^2 value given parameter values. This function is minimized with iminuit"
        diff = self.y - self._pdf(self.x, *args)
        return (diff**2).sum()
    
    def r2(self,ncall=None):
        start = [p.start for p in self.fit_params]
        limits = [(p.min, p.max) for p in self.fit_params]
        
        self.minimizer = iminuit.Minuit(self._r2, *start)
        self.minimizer.limits = limits
        ret = self.minimizer.migrad(ncall)

        for i in range(len(self.fit_params)):
            self.fit_params[i].value = self.minimizer.values[i]
            self.fit_params[i].error = self.minimizer.errors[i]

        return ret

    def MLE(self, ncall=None):
        start = [p.start for p in self.fit_params]
        limits = [(p.min, p.max) for p in self.fit_params]
        
        self.minimizer = iminuit.Minuit(cost.UnbinnedNLL(self.arr, self._pdf), *start)
        self.minimizer.limits = limits
        ret = self.minimizer.migrad(ncall)

        for i in range(len(self.fit_params)):
            self.fit_params[i].value = self.minimizer.values[i]
            self.fit_params[i].error = self.minimizer.errors[i]
        return ret
    
    def values(self):
        "Returns the fit values in the structure of the pdf array"
        return [ {list(signature(pdf.fcn).parameters.keys())[i+1]: pdf._params[i].value for i in range(len(pdf._params))} for pdf in self.pdf]

    def errors(self):
        "Returns the fit errors in the structure of the pdf array"
        return [ {list(signature(pdf.fcn).parameters.keys())[i+1]: pdf._params[i].error for i in range(len(pdf._params))} for pdf in self.pdf]

    def summary(self):
        """
        Print a summary table of the parameter estimates.
        """
        component_data = []

        for component_idx, pdf in enumerate(self.pdf):
            params = []
            param_names = list(signature(pdf.fcn).parameters.keys())[1:]
            
            for i, name in enumerate(param_names):
                values = pdf._params[i].value
                errors = pdf._params[i].error

                if isinstance(values, list) and isinstance(errors, list):
                    # Handle list of values and errors
                    for idx, (value, error) in enumerate(zip(flatten(values),flatten(errors))):
                        params.append({
                            "Parameter": f"{name}[{idx}]",
                            "Value": value,
                            "Error": error,
                        })
                else:
                    # Single value and error
                    params.append({
                        "Parameter": name,
                        "Value": values,
                        "Error": errors,
                    })

            component_data.append((f"Component {component_idx + 1}", params))

        # Print each component's data
        for component_name, params in component_data:
            print(component_name)
            print(tabulate(
                params,
                headers="keys",
                floatfmt=".6g",
                tablefmt="github"
            ))
            print()

    def plot_data(self):
        """Plots the histogram centers and entries of the data"""
        plt.errorbar(self.x, self.y, yerr = self.err_b, linestyle='', fmt='.', ecolor='black', color='black',elinewidth=1, capsize=0 )
    
    def plot_fit(self):
        """Plots a curve of the function estimate"""
        x = np.linspace(self.range[0],self.range[1],1000)
        plt.plot(x, self._pdf(x) * self.widths.mean())

