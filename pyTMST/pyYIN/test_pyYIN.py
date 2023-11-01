import unittest
import os
import argparse

import matlab.engine
import numpy as np
import soundfile as sf
from scipy.signal import hilbert, butter, lfilter

from . import mock_yin


MATLAB_TOOLBOX_PATHS = [
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


    def test_mock_yin(self):
        audio_path = './LaVoixHumaine_6s.wav'
        f0_min, f0_max = 100, 500
        thresh = 0.1
        hop = 20

        sig, fs = sf.read(audio_path)
        sig = sig[:,0]
        py_f0, py_ap0 = mock_yin(sig, fs, f0_min, f0_max, thresh, hop)

        self.eng.eval(f"[sig, fs] = audioread('{audio_path}');", nargout=0)
        self.eng.eval(f"P = struct('sr', {fs}, 'minf0', {f0_min}, 'maxf0', {f0_max}, 'hop', {hop}, 'thresh', {thresh});", nargout=0)
        self.eng.eval(f"r = yin(sig(:,1), P);", nargout=0)
        mat_f0 = np.squeeze(self.eng.workspace['r']['f0'])
        mat_ap0 = np.squeeze(self.eng.workspace['r']['ap0'])

        np.testing.assert_allclose(mat_f0, py_f0,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_ap0, py_ap0,
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

