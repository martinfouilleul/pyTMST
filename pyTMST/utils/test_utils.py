import unittest
import os
import argparse

import matlab.engine
import numpy as np
import soundfile as sf
from scipy.signal import hilbert, butter, lfilter

from . import utils


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eng = matlab.engine.start_matlab()

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

