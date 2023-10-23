from collections import namedtuple

import numpy as np
from scipy.signal import hilbert

from .utils import define_modulation_axis, periodogram
from .pyLTFAT import erb_filt_bw
from .pyAMT import apply_gammatone_filterbank, king2019_modfilterbank_updated


AMa_spec_params = namedtuple('AMa_spec_params', ['t', 'f_bw', 'gamma_responses', 'E', 'mf', 'mfb'])
AMi_spec_params = namedtuple('AMi_spec_params', ['t', 'f_bw', 'gamma_responses', 'E', 'mf', 'AMrms', 'DC'])


def AMa_spectrum(sig, fs, mfmin=0.5, mfmax=200, modbank_Nmod=200, fmin=70, fmax=6700):
    if not isinstance(sig, np.ndarray) or not isinstance(fs, (int, float)):
        raise ValueError("Invalid input types.")
    if fs <= 0:
        raise ValueError("fs must be a positive scalar.")
    
    t = np.arange(1,len(sig)+1) / fs
    gamma_responses, fc = apply_gammatone_filterbank(sig, fs, fmin, fmax)
    E = np.abs(hilbert(gamma_responses, axis=1))
    
    f_spectra, f_spectra_intervals = define_modulation_axis(mfmin, mfmax, modbank_Nmod)
    Nchan = fc.shape[0]
    AMspec = np.zeros((f_spectra.shape[0], Nchan))
    for ichan in range(Nchan):
        Pxx = periodogram(E[ichan, :], fs, f_spectra)
        AMspec[:, ichan] = 2 * Pxx
   
    step = AMa_spec_params(t, erb_filt_bw(fc), gamma_responses, E, f_spectra, f_spectra_intervals)
    return AMspec, fc, f_spectra, step


def AMi_spectrum(sig, fs, mfmin=0.5, mfmax=200., modbank_Nmod=200, modbank_Qfactor=1, fmin=70, fmax=6700):
    if not isinstance(sig, np.ndarray) or not isinstance(fs, (int, float)):
        raise ValueError("Invalid input types.")
    if fs <= 0:
        raise ValueError("fs must be a positive scalar.")
    
    t = np.arange(1,len(sig)+1) / fs
    gamma_responses, fc = apply_gammatone_filterbank(sig, fs, fmin, fmax)
    E = np.abs(hilbert(gamma_responses, axis=1))

    Nchan = fc.shape[0]
    AMfilt, mf, _ = king2019_modfilterbank_updated(E.T, fs, mfmin, mfmax, modbank_Nmod, modbank_Qfactor)

    AMrms = np.sqrt(np.mean(AMfilt ** 2, axis=2)) * np.sqrt(2)
    DC = np.mean(E.T, axis=0)
    AMIspec = AMrms.T / (DC[:, np.newaxis] * np.ones(mf.shape[0]))

    step = AMi_spec_params(t, erb_filt_bw(fc), gamma_responses, E, mf, AMrms, DC)
    return AMIspec, fc, mf, step

