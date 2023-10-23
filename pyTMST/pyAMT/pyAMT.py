"""
Author: Anton Zickler
Copyright (c) 2023 A. Zickler, M. Ernst, L. Varnet, A. Tavano

Based on the [Auditory Modeling Toolbox (AMT)](https://amtoolbox.org/) for
MATLAB:
 - Majdak, P., Hollomey, C., and Baumgartner, R. (2022). "AMT 1.x: A toolbox
   for reproducible research in auditory modeling," Acta Acustica 6:19
   https://doi.org/10.1051/aacus/2022011
 - Søndergaard, P. and Majdak, P. (2013). "The Auditory Modeling Toolbox," in
   The Technology of Binaural Listening, edited by Blauert, J. (Springer,
   Berlin, Heidelberg), pp. 33-56

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
from  scipy.signal import butter, lfilter
from gammatone.filters import make_erb_filters, erb_filterbank

from ..pyLTFAT import aud_space_bw


def auditory_filterbank(sig, fs, fmin, fmax):
    """
    Authors of the original MATLAB code: Peter L. Søndergaard
    """

    bw = 1
    fc = aud_space_bw(fmin, fmax, bw)
    erb_coeff_arr = make_erb_filters(fs, fc)
    gamma_responses = erb_filterbank(sig, erb_coeff_arr)

    return gamma_responses, fc


def king2019_modfilterbank_updated(sig, fs, mfmin, mfmax, modbank_Nmod, modbank_Qfactor):
    """
    Authors of the original MATLAB code:
    - Leo Varnet and Andrew King (2020)
    - Alejandro Osses (2020) Original implementation for the AMT
    - Clara Hollomey (2021) Adapted for AMT
    - Piotr Majdak (2021) Further adaptations to AMT 1.0
    """

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

