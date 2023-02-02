import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
#from multiprocessing import cpu_count
#from multiprocessing import Pool

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []
    
    # Estimate distribution parameters from data
    def mapping(distribution):

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                return (distribution, params, sse, pd.Series(pdf))

        except Exception:
            return None 
    
    distros=[d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]
    
    fitted_distros=list(map(mapping,distros))
    
    
    for distro in fitted_distros:
        if distro: best_distributions+=[distro]
    
    
    return sorted(best_distributions, key=lambda x:x[2]) 

def make_pdf(dist, params, size=int(2e6)):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(1e-8, *arg, loc=loc, scale=scale) if arg else dist.ppf(1e-8, loc=loc, scale=scale)
    end = dist.ppf(1.0-1e-9, *arg, loc=loc, scale=scale) if arg else dist.ppf(1.0-1e-9, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf
