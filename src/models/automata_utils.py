from src.models.paa import paa_from_config
from src.models.sax import sax_from_config


def convert_series_to_symbols(series, config=None):
    paa_values = paa_from_config(series, config=config)
    sax_string = sax_from_config(paa_values, config=config)
    return sax_string
