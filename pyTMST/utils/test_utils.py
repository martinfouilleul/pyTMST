"""
Copyright (c) 2023 A. Zickler, A. Tavano, L. Varnet

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License (CC BY-NC 4.0).
You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc/4.0/>.
"""


import unittest
import os
import argparse

import matlab.engine
import numpy as np
import soundfile as sf
from scipy.signal import hilbert, butter, lfilter

from . import utils


MATLAB_TOOLBOX_PATHS = [
    "./matlab_toolboxes/TMST/",
    "./matlab_toolboxes/amtoolbox",
    "./matlab_toolboxes/yin"
    ]


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eng = matlab.engine.start_matlab()

        for path in MATLAB_TOOLBOX_PATHS:
            if os.path.exists(path):
                cls.eng.eval(f"addpath(genpath('{path}'))", nargout=0)
            else:
                print(f"The path {path} does not exist.")

    @classmethod
    def tearDownClass(cls):
        cls.eng.quit()


    def test_define_modulation_axis(self):
        mfmin, mfmax = 0.5, 200
        nf = 200

        py_f_spectra, py_f_spectra_intervals = utils.define_modulation_axis(mfmin, mfmax, nf)

        self.eng.eval(f"f_spectra_intervals = logspace(log10({mfmin}), log10({mfmax}), {nf}+1);", nargout=0)
        self.eng.eval(f"f_spectra = logspace(log10(sqrt(f_spectra_intervals(1)*f_spectra_intervals(2))), log10(sqrt(f_spectra_intervals(end)*f_spectra_intervals(end-1))), {nf});", nargout=0)
        mat_f_spectra = np.squeeze(self.eng.workspace['f_spectra'])
        mat_f_spectra_intervals = np.squeeze(self.eng.workspace['f_spectra_intervals'])

        np.testing.assert_allclose(mat_f_spectra, py_f_spectra,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_f_spectra_intervals, py_f_spectra_intervals,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_gausswin(self):
        n = 100
        alpha = 2.5

        py_gwin = utils.gausswin(n, alpha)

        self.eng.eval(f"gwin = gausswin({n}, {alpha});", nargout=0)
        mat_gwin = np.squeeze(self.eng.workspace['gwin'])

        np.testing.assert_allclose(mat_gwin, py_gwin,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_segment_into_windows(self):
        audio_path = './LaVoixHumaine_6s.wav'
        sig, fs = sf.read(audio_path)
        sig = sig[:,0]

        width = 4 * (1/fs)
        shift = 0.1
        gwin = False

        py_windows = utils.segment_into_windows(sig, fs, width, shift, gwin)

        self.eng.eval(f"[sig, fs] = audioread('{audio_path}');", nargout=0)
        self.eng.eval(f"windows = windowing(sig(:,1), fs, {width}, {shift}, {int(gwin)});", nargout=0)
        mat_windows = np.squeeze(self.eng.workspace['windows']).T

        np.testing.assert_allclose(mat_windows, py_windows,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_periodogram(self):
        fs = 1000
        t = np.arange(0, 1, 1/fs)
        sig = np.cos(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t)
        freqs = [100, 200]

        py_pxx = utils.periodogram(sig, fs, freqs)

        self.eng.eval(f"[pxx, f] = periodogram({matlab.double(sig.tolist())},[],{freqs},{fs},'psd');", nargout=0)
        mat_pxx = np.squeeze(self.eng.workspace['pxx'])

        np.testing.assert_allclose(mat_pxx, py_pxx,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_lombscargle(self):
        def random_monotonic_range(a, b, size):
            rand_nums = np.random.rand(size)
            rand_nums /= np.sum(rand_nums)
            cumul_nums = np.cumsum(rand_nums)
            return a + (b - a) * cumul_nums

        audio_path = './LaVoixHumaine_6s.wav'
        freqs = np.linspace(0.01, 10, 1000)

        sig, fs = sf.read(audio_path)
        sig = sig[:,0]
        sig[::5] = np.nan
        t = random_monotonic_range(0., len(sig) / fs, len(sig))
        py_f, py_pxx = utils.lombscargle(t, sig, freqs)

        self.eng.workspace['freqs'] = matlab.double(freqs.tolist())
        self.eng.workspace['t'] = matlab.double(t.tolist())
        self.eng.workspace['sig'] = matlab.double(sig.tolist())
        self.eng.eval(f"[pxx, f] = plomb(sig, t, freqs);", nargout=0)
        mat_f = np.squeeze(self.eng.workspace['f'])
        mat_pxx = np.squeeze(self.eng.workspace['pxx'])

        np.testing.assert_allclose(mat_f, py_f,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_pxx, py_pxx,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_interpmean(self):
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([0, 1, np.nan, 3, 4, 5])
        xi = np.array([0, 2, 3.5, 4])

        py_result = utils.interpmean(x, y, xi)

        self.eng.workspace['x'] = matlab.double(x.tolist())
        self.eng.workspace['y'] = matlab.double(y.tolist())
        self.eng.workspace['xi'] = matlab.double(xi.tolist())
        self.eng.eval(f"result = interpmean(x, y, xi);", nargout=0)
        mat_result = np.squeeze(self.eng.workspace['result'])

        np.testing.assert_allclose(mat_result, py_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_get_non_nan_segments(self):
        arr = np.array([np.nan, 1, 2, 3, np.nan, np.nan, 4, 5, np.nan])
        self.assertEqual(utils.get_non_nan_segments(arr), [(1, 4), (6, 8)])


    def test_filter_max_jump(self):
        arr = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        maxjump = 5.0
        expected_output = np.array([1.0, 2.0, 4.0, np.nan, 16.0])
        np.testing.assert_array_equal(utils.filter_max_jump(arr, maxjump), expected_output)


    def test_filter_by_duration(self):
        arr = np.array([np.nan, 1, 2, 3, np.nan, np.nan, 4, 5, np.nan])
        filtered_arr = utils.filter_by_duration(arr, fs=1, minduration=3)
        expected_filtered_arr = np.array([np.nan, 1, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(filtered_arr, expected_filtered_arr)


    def test_filter_by_variability(self):
        arr = np.array([np.nan, 1, 1, 1, np.nan, np.nan, 4, 5, np.nan])
        filtered_arr = utils.filter_by_variability(arr, fs=1, var_thres=0.1)
        expected_filtered_arr = np.array([np.nan, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(filtered_arr, expected_filtered_arr)


    def test_filter_by_absolute_range(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        minf = 2.0
        maxf = 4.0
        expected_output = np.array([np.nan, 2.0, 3.0, 4.0, np.nan])
        result = utils.filter_by_absolute_range(arr, minf, maxf)
        np.testing.assert_array_equal(result, expected_output)


    def test_filter_by_relative_range(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        frange_median = [0.5, 1.5]
        expected_output = np.array([np.nan, 2.0, 3.0, 4.0, np.nan])
        result = utils.filter_by_relative_range(arr, frange_median)
        np.testing.assert_array_equal(result, expected_output)


    def test_remove_artifacts(self):
        sig = np.array([np.nan, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1.0, 1.0, 1.0, np.nan])

        fs = 1
        max_jump = 5.0
        min_duration = 2
        f_range = [1.0, 20.0]
        var_thresh = 0.5
        f_range_median = [0.5, 1.5]

        py_result = utils.remove_artifacts(
            sig,
            fs=fs,
            max_jump=max_jump,
            min_duration=min_duration,
            f_range=f_range,
            f_range_median=f_range_median,
            var_thresh=var_thresh
        )

        mat_result = self.eng.remove_artifacts_FM(
            matlab.double(sig.tolist()),
            fs,
            max_jump,
            min_duration,
            matlab.double(f_range),
            matlab.double(f_range_median),
            var_thresh
        )

        np.testing.assert_array_equal(np.squeeze(mat_result), py_result)


    def test_hilbert(self):
        t = np.arange(0, 1 + 1/1024, 1/1024)
        f = 60
        x = np.sin(2 * np.pi * f * t)
        matlab_result = np.squeeze(self.eng.hilbert(x))
        python_result = hilbert(x)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_lp_butter_filter(self):
        audio_path = './LaVoixHumaine_6s.wav'
        order = 1
        cutoff = 150

        sig, fs = sf.read(audio_path)
        sig = sig[:,0]
        py_b, py_a = butter(order, cutoff, fs=fs)
        py_outsig = lfilter(py_b, py_a, sig)
        
        self.eng.eval(f"[sig, fs] = audioread('{audio_path}');", nargout=0)
        self.eng.eval(f"[b, a] = butter({order}, {cutoff}/(fs/2));", nargout=0)
        self.eng.eval("outsig = filter(b, a, sig(:,1));", nargout=0)
        mat_b = np.squeeze(self.eng.workspace['b'])
        mat_a = np.squeeze(self.eng.workspace['a'])
        mat_outsig = np.squeeze(self.eng.workspace['outsig'])

        np.testing.assert_allclose(mat_b, py_b,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_a, py_a,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_outsig, py_outsig,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unit tests for the pyTMST.utils module.')
    parser.add_argument('--float_rel_tolerance', type=float, default=1.e-7,
                        help='Relative tolerance for floating point comparisons')
    parser.add_argument('--float_abs_tolerance', type=float, default=0,
                        help='Absolute tolerance for floating point comparisons')
    parser.add_argument('--verbosity', type=int, choices=[0,1,2], default=0,
                        help='Level of verbosity for test output')

    args = parser.parse_args()

    TestUtils.float_rel_tolerance = args.float_rel_tolerance
    TestUtils.float_abs_tolerance = args.float_abs_tolerance
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

