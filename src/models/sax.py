import numpy as np
from scipy.stats import norm


def compute_breakpoints(alphabet_size):
    quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    breakpoints = norm.ppf(quantiles)
    return breakpoints
