"""
Copyright (c) 2023 A. Zickler, A. Tavano, L. Varnet

Based on the [Large Time-Frequency Analysis Toolbox (LTFAT)](https://ltfat.org)
for MATLAB: 
 - Zdeněk Průša, Peter L. Søndergaard, Nicki Holighaus, Christoph Wiesmeyr,
   Peter Balazs The Large Time-Frequency Analysis Toolbox 2.0. Sound, Music,
   and Motion, Lecture Notes in Computer Science 2014, pp 419-442
 - Peter L. Søndergaard, Bruno Torrésani, Peter Balazs. The Linear
   Time-Frequency Analysis Toolbox. International Journal of Wavelets,
   Multiresolution Analysis and Information Processing, 10(4), 2012.

Author of the MATLAB functions that the code in this file emulate: Peter L.
Søndergaard

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


def freq_to_aud(freq):
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def aud_to_freq(aud):
    return (1 / 0.00437) * np.sign(aud) * (np.exp(np.abs(aud) / 9.2645) - 1)


def aud_filt_bw(fc):
    return 24.7 + fc/9.265;


def aud_space_bw(fmin, fmax, bw=1.):
    if fmin < 0 or fmax < 0 or fmin > fmax:
        raise ValueError("Invalid frequency bounds. Make sure 0 <= fmin <= fmax.")
    
    if bw <= 0:
        raise ValueError("Bandwidth (bw) must be a positive scalar.")

    aud_limits = freq_to_aud(np.array([fmin, fmax]))
    aud_range = aud_limits[1] - aud_limits[0]

    n = int(np.floor(aud_range / bw))
    remainder = aud_range - n * bw

    aud_points = aud_limits[0] + np.arange(0, n + 1) * bw + remainder / 2
    
    y = aud_to_freq(aud_points)
    return y

