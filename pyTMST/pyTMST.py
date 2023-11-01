"""
Author: Anton Zickler
Copyright (c) 2023 A. Zickler, M. Ernst, L. Varnet, A. Tavano

Python port of the [Temporal Modulation Spectrum Toolbox
(TMST)](https://github.com/LeoVarnet/TMST) for MATLAB:
> L. Varnet (2023). "Temporal Modulation Spectrum Toolbox: A Matlab toolbox
> for the computation of amplitude- and f0- modulation spectra and
> spectrograms."

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License (CC BY-NC 4.0).
You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc/4.0/>.
"""


from collections import namedtuple

import numpy as np
from scipy.signal import hilbert

from .utils import define_modulation_axis, periodogram, lombscargle, remove_artifacts, interpmean
from .pyLTFAT import aud_filt_bw
from .pyAMT import auditory_filterbank, king2019_modfilterbank_updated
from .pyYIN import mock_yin


AMa_spec_params = namedtuple('AMa_spec_params', ['t', 'f_bw', 'gamma_responses', 'E', 'mf', 'mfb'])
AMi_spec_params = namedtuple('AMi_spec_params', ['t', 'f_bw', 'gamma_responses', 'E', 'mf', 'AMrms', 'DC'])
f0M_spec_params = namedtuple('f0M_spec_params', ['t', 'f0', 'mf', 'mfb'])


def AMa_spectrum(sig, fs, mfmin=0.5, mfmax=200, modbank_Nmod=200, fmin=70, fmax=6700):
    if not isinstance(sig, np.ndarray) or not isinstance(fs, (int, float)):
        raise ValueError("Invalid input types.")
    if fs <= 0:
        raise ValueError("fs must be a positive scalar.")
    
    t = np.arange(1,len(sig)+1) / fs
    gamma_responses, fc = auditory_filterbank(sig, fs, fmin, fmax)
    E = np.abs(hilbert(gamma_responses, axis=1))
    
    f_spectra, f_spectra_intervals = define_modulation_axis(mfmin, mfmax, modbank_Nmod)
    Nchan = fc.shape[0]
    AMspec = np.zeros((f_spectra.shape[0], Nchan))
    for ichan in range(Nchan):
        Pxx = periodogram(E[ichan, :], fs, f_spectra)
        AMspec[:, ichan] = 2 * Pxx
   
    step = AMa_spec_params(t, aud_filt_bw(fc), gamma_responses, E, f_spectra, f_spectra_intervals)
    return AMspec, fc, f_spectra, step


def AMi_spectrum(sig, fs, mfmin=0.5, mfmax=200., modbank_Nmod=200, modbank_Qfactor=1, fmin=70, fmax=6700):
    if not isinstance(sig, np.ndarray) or not isinstance(fs, (int, float)):
        raise ValueError("Invalid input types.")
    if fs <= 0:
        raise ValueError("fs must be a positive scalar.")
    
    t = np.arange(1,len(sig)+1) / fs
    gamma_responses, fc = auditory_filterbank(sig, fs, fmin, fmax)
    E = np.abs(hilbert(gamma_responses, axis=1))

    Nchan = fc.shape[0]
    AMfilt, mf, _ = king2019_modfilterbank_updated(E.T, fs, mfmin, mfmax, modbank_Nmod, modbank_Qfactor)

    AMrms = np.sqrt(np.mean(AMfilt ** 2, axis=2)) * np.sqrt(2)
    DC = np.mean(E.T, axis=0)
    AMIspec = AMrms.T / (DC[:, np.newaxis] * np.ones(mf.shape[0]))

    step = AMi_spec_params(t, aud_filt_bw(fc), gamma_responses, E, mf, AMrms, DC)
    return AMIspec, fc, mf, step


def f0M_spectrum(sig, fs, mfmin=.5, mfmax=200., modbank_Nmod=200, undersample=20, fmin=60, fmax=550, yin_thresh=.2, ap0_thresh=.8, max_jump=10, min_duration=.08):
    w_len = -(fs // -fmin) # ceiling division
    f0, ap0 = mock_yin(sig, fs, fmin, fmax, yin_thresh, undersample)

    f0 = 440 * np.power(2, f0)
    f0[ap0 > ap0_thresh] = np.nan
    f0 = remove_artifacts(f0, fs/undersample, max_jump, min_duration, (fmin, fmax), (.4, 2.5), 1500)
    f0_wo_nan = f0[~np.isnan(f0)]

    t = np.arange(1, len(sig) + 1) / fs
    t_wo_nan = t[::undersample]
    t_wo_nan = t_wo_nan[:len(f0)]
    t_wo_nan = t_wo_nan[~np.isnan(f0)]

    f_spectra, f_spectra_intervals = define_modulation_axis(mfmin, mfmax, modbank_Nmod)

    _, f0Mfft = lombscargle(t_wo_nan, f0_wo_nan, f_spectra)
    f0Mfft *= 2
    f0M_spectrum = interpmean(f_spectra, f0Mfft, f_spectra_intervals)

    t_f0 = np.arange(1, len(f0) + 1) / (fs / undersample)
    step = f0M_spec_params(t_f0, f0, f_spectra, f_spectra_intervals)

    return f0M_spectrum, f_spectra, step

