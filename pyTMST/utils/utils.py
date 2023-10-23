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

