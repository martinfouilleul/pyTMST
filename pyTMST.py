from collections import namedtuple

import numpy as np
import scipy.signal
import gammatone.filters



AMa_spec_params = namedtuple('AMa_spec_params', ['t', 'f_bw', 'gamma_responses', 'E', 'mf', 'mfb'])


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


def apply_gammatone_filterbank(sig, fs, fmin, fmax):
    bw = 1
    fc = erbspace_bw(fmin, fmax, bw)
    erb_coeff_arr = gammatone.filters.make_erb_filters(fs, fc)
    gamma_responses = gammatone.filters.erb_filterbank(sig, erb_coeff_arr)
    return gamma_responses, fc


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


def AMa_spectrum(sig, fs, mfmin=0.5, mfmax=200, modbank_Nmod=200, fmin=70, fmax=6700):
    if not isinstance(sig, np.ndarray) or not isinstance(fs, (int, float)):
        raise ValueError("Invalid input types.")
    if fs <= 0:
        raise ValueError("fs must be a positive scalar.")
    
    t = np.arange(1,len(sig)+1) / fs
    gamma_responses, fc = apply_gammatone_filterbank(sig, fs, fmin, fmax)
    E = np.abs(scipy.signal.hilbert(gamma_responses, axis=0))
    
    f_spectra, f_spectra_intervals = define_modulation_axis(mfmin, mfmax, modbank_Nmod)
    Nchan = fc.shape[0]
    AMspec = np.zeros((f_spectra.shape[0], Nchan))
    for ichan in range(Nchan):
        Pxx = periodogram(E[:, ichan], fs, f_spectra)
        AMspec[:, ichan] = 2 * Pxx
   
    step = AMa_spec_params(t, erb_filt_bw(fc), gamma_responses, E, f_spectra, f_spectra_intervals)
    return AMspec, fc, f_spectra, step

