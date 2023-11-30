"""
Author: Anton Zickler
Copyright (c) 2023 A. Zickler, M. Ernst, L. Varnet, A. Tavano

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License (CC BY-NC 4.0).
You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc/4.0/>.
"""


import numpy as np


def define_modulation_axis(mfmin, mfmax, nf):
    f_spectra_intervals = np.logspace(np.log10(mfmin), np.log10(mfmax), nf + 1)
    f_spectra = np.logspace(np.log10(np.sqrt(f_spectra_intervals[0] * f_spectra_intervals[1])),
                            np.log10(np.sqrt(f_spectra_intervals[-1] * f_spectra_intervals[-2])), nf)
    return f_spectra, f_spectra_intervals


def gausswin(N, alpha=2.5):
    n = np.arange(N)
    window = np.exp(-0.5 * ((alpha * (n - (N - 1) / 2)) / ((N - 1) / 2)) ** 2)
    return window


def segment_into_windows(signal, fs, width, shift, gwin):
    signal = np.array(signal).flatten()
    width_in_sample = int(np.floor(width * fs))
    shift_in_sample = int(np.floor(shift * fs))

    if gwin:
        G = gausswin(width_in_sample)
        norm = 1
    else:
        G = np.ones(width_in_sample)
        norm = 1

    windows = []
    i = 0
    while i + width_in_sample < len(signal):
        windowed_signal = (signal[i:i+width_in_sample] * G) / norm
        windows.append(windowed_signal)
        i += shift_in_sample

    return np.array(windows)


def periodogram(sig, fs, freqs):
    N = np.array(sig).shape[0]
    T = 1.0 / fs
    dt = np.arange(0, N) * T
    
    dt = dt[:, np.newaxis]
    freqs = np.array(freqs)[np.newaxis, :]
    sig_expanded = sig[:, np.newaxis]
    
    X_f = np.sum(sig_expanded * np.exp(-1j * 2 * np.pi * freqs * dt), axis=0)
    
    psd = (np.abs(X_f)**2) / (N * fs)
    return psd

