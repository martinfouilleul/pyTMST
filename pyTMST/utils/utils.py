"""
Author: Anton Zickler
Copyright (c) 2023 A. Zickler, M. Ernst, L. Varnet, A. Tavano

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License (CC BY-NC 4.0).
You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc/4.0/>.
"""


import numpy as np
import scipy.signal


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


def lombscargle(t, sig, f):
    avg_fs = np.mean(np.diff(t))**-1
    pxx = (2 / avg_fs) * scipy.signal.lombscargle(t, sig - np.mean(sig), 2*np.pi*f)
    return f, pxx


def interpmean(x, y, xi):
    Ni = len(xi)
    yi = np.empty(Ni - 1)
    yi.fill(np.nan)

    for i_sample in range(Ni - 1):
        idx = (x >= xi[i_sample]) & (x <= xi[i_sample + 1])
        yi[i_sample] = np.nanmean(y[idx])

    return yi


def get_non_nan_segments(arr):
    is_not_nan = ~np.isnan(arr)
    segments = []
    i = 0
    while i < len(is_not_nan):
        if is_not_nan[i]:
            start = i
            while i < len(is_not_nan) and is_not_nan[i]:
                i += 1
            end = i
            segments.append((start, end))
        i += 1
    return segments


def filter_max_jump(arr, maxjump):
    arr_copy = np.copy(arr)
    diff_arr = np.diff(arr_copy)
    arr_copy[:-1][np.abs(diff_arr) > maxjump] = np.nan
    return arr_copy


def filter_by_duration(arr, fs, minduration):
    arr_filtered = arr.copy()
    nminsamples = minduration * fs
    segments = get_non_nan_segments(arr_filtered)
    nan_indices = np.zeros_like(arr_filtered, dtype=bool)
    for (start, end) in segments:
        if end - start < nminsamples:
            nan_indices[start:end] = True
    arr_filtered[nan_indices] = np.nan
    return arr_filtered


def filter_by_variability(arr, fs, var_thres):
    arr_filtered = arr.copy()
    segments = get_non_nan_segments(arr_filtered)
    nan_indices = np.zeros_like(arr_filtered, dtype=bool)
    for (start, end) in segments:
        segment = arr_filtered[start:end]
        segment_var = np.nansum(np.abs(np.diff(segment))) / ((end - start) / fs)
        if segment_var > var_thres:
            nan_indices[start:end] = True
    arr_filtered[nan_indices] = np.nan
    return arr_filtered


def filter_by_absolute_range(arr, minf, maxf):
    new_array = arr.copy()
    new_array[(new_array < minf) | (new_array > maxf)] = np.nan
    return new_array


def filter_by_relative_range(arr, f_range_median):
    new_array = arr.copy()
    median_arr = np.nanmedian(arr)
    new_array[(new_array < f_range_median[0] * median_arr) | (new_array > f_range_median[1] * median_arr)] = np.nan
    return new_array


def remove_artifacts(sig, fs, max_jump, min_duration, f_range, f_range_median=None, var_thresh=0):
    filtered_sig = np.copy(sig)
    filtered_sig = filter_max_jump(filtered_sig, max_jump)
    filtered_sig = filter_by_duration(filtered_sig, fs, min_duration)
    filtered_sig = filter_by_absolute_range(filtered_sig, f_range[0], f_range[1])
    filtered_sig = filter_by_variability(filtered_sig, fs, var_thresh)
    if f_range_median is not None:
        filtered_sig = filter_by_relative_range(filtered_sig, f_range_median)
    return filtered_sig

