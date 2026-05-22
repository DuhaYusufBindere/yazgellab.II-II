import numpy as np


def paa_transform(series, window_size):
    n = len(series)
    num_segments = n // window_size
    trimmed = series[:num_segments * window_size]
    segments = trimmed.reshape(num_segments, window_size)
    return segments.mean(axis=1)
