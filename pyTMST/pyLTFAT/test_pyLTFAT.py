"""
Author: Anton Zickler
Copyright (c) 2023 A. Zickler, M. Ernst, L. Varnet, A. Tavano

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


import unittest
import os
import argparse

import matlab.engine
import numpy as np

from . import pyLTFAT


MATLAB_LTFAT_PATH = "./matlab_toolboxes/amtoolbox/thirdparty/ltfat/"


class TestPyLTFAT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eng = matlab.engine.start_matlab()

        if os.path.exists(MATLAB_LTFAT_PATH):
            cls.eng.eval(f"addpath(genpath('{MATLAB_LTFAT_PATH}'))", nargout=0)
        else:
            print(f"The path {MATLAB_LTFAT_PATH} does not exist.")

    @classmethod
    def tearDownClass(cls):
        cls.eng.quit()


    def test_freq_to_aud(self):
        freq = 70.;
        matlab_result = self.eng.freqtoaud(freq)
        python_result = pyLTFAT.freq_to_aud(freq)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_aud_to_freq(self):
        erb = 31.6
        matlab_result = self.eng.audtofreq(erb)
        python_result = pyLTFAT.aud_to_freq(erb)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

    def test_aud_filt_bw(self):
        fc = 70.
        mat_bw = self.eng.audfiltbw(fc)
        py_bw = pyLTFAT.aud_filt_bw(fc)
        np.testing.assert_allclose(mat_bw, py_bw,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

    def test_aud_space_bw(self):
        fmin, fmax = 70., 6700.
        matlab_result = np.array(self.eng.audspacebw(fmin, fmax)).squeeze()
        python_result = pyLTFAT.aud_space_bw(fmin, fmax, bw=1)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unit tests for the pyLTFAT module.')
    parser.add_argument('--float_rel_tolerance', type=float, default=1.e-7,
                        help='Relative tolerance for floating point comparisons')
    parser.add_argument('--float_abs_tolerance', type=float, default=0,
                        help='Absolute tolerance for floating point comparisons')
    parser.add_argument('--verbosity', type=int, choices=[0,1,2], default=0,
                        help='Level of verbosity for test output')

    args = parser.parse_args()

    TestPyLTFAT.float_rel_tolerance = args.float_rel_tolerance
    TestPyLTFAT.float_abs_tolerance = args.float_abs_tolerance
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPyLTFAT)
    unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

