import numpy as np
from  scipy.signal import butter, lfilter
from gammatone.filters import make_erb_filters, erb_filterbank

from .. import pyLTFAT


def apply_gammatone_filterbank(sig, fs, fmin, fmax):
    bw = 1
    fc = pyLTFAT.erbspace_bw(fmin, fmax, bw)
    erb_coeff_arr = make_erb_filters(fs, fc)
    gamma_responses = erb_filterbank(sig, erb_coeff_arr)
    return gamma_responses, fc


def king2019_modfilterbank_updated(sig, fs, mfmin, mfmax, modbank_Nmod, modbank_Qfactor):
    logfmc = np.linspace(np.log(mfmin), np.log(mfmax), modbank_Nmod)
    mfc = np.exp(logfmc);

    flim = np.zeros((modbank_Nmod, 2))
    b = np.zeros((modbank_Nmod, 3))
    a = np.zeros((modbank_Nmod, 3))
    outsig = np.zeros((sig.shape[0], sig.shape[1], modbank_Nmod))

    for ichan in range(modbank_Nmod):
        flim[ichan, :] = mfc[ichan] * np.sqrt(4 + 1 / modbank_Qfactor ** 2) / 2 + np.array([-1, 1]) * mfc[ichan] / modbank_Qfactor / 2
        b[ichan, :], a[ichan, :] = butter(1, 2 * flim[ichan, :] / fs, btype='band')
        outsig[:, :, ichan] = lfilter(b[ichan, :], a[ichan, :], sig, axis=0)

    step = {
        'b': b,
        'a': a,
    }

    return outsig.T, mfc, step

