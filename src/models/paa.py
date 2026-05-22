import numpy as np


def paa_transform(series, window_size):
    n = len(series)
    num_segments = n // window_size
    trimmed = series[:num_segments * window_size]
    segments = trimmed.reshape(num_segments, window_size)
    return segments.mean(axis=1)


def paa_from_config(series, config=None):
    from src.config.config_loader import load_config
    if config is None:
        config = load_config()
    window_size = config.get("automata", {}).get("default_window_size", 4)
    return paa_transform(series, window_size)
