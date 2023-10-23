import numpy as np


def freq_to_erb(freq):
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb_to_freq(erb):
    return (1 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erb_filt_bw(fc):
    return 24.7 + fc/9.265;


def erbspace_bw(fmin, fmax, bw=1.):
    if fmin < 0 or fmax < 0 or fmin > fmax:
        raise ValueError("Invalid frequency bounds. Make sure 0 <= fmin <= fmax.")
    
    if bw <= 0:
        raise ValueError("Bandwidth (bw) must be a positive scalar.")

    erb_limits = freq_to_erb(np.array([fmin, fmax]))
    erb_range = erb_limits[1] - erb_limits[0]

    n = int(np.floor(erb_range / bw))
    remainder = erb_range - n * bw

    erb_points = erb_limits[0] + np.arange(0, n + 1) * bw + remainder / 2
    
    y = erb_to_freq(erb_points)
    return y

