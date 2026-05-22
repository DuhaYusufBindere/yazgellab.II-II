import numpy as np
from scipy.stats import norm


def compute_breakpoints(alphabet_size):
    quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    breakpoints = norm.ppf(quantiles)
    return breakpoints


def sax_transform(paa_values, alphabet_size):
    breakpoints = compute_breakpoints(alphabet_size)
    symbols = []
    
    for val in paa_values:
        # val'ın düştüğü breakpointi bul
        idx = np.searchsorted(breakpoints, val)
        symbol = chr(97 + idx) 
        symbols.append(symbol)
        
    return "".join(symbols)


def sax_from_config(paa_values, config=None):
    from src.config.config_loader import load_config
    if config is None:
        config = load_config()
    alphabet_size = config.get("automata", {}).get("default_alphabet_size", 3)
    return sax_transform(paa_values, alphabet_size)
