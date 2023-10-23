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
import soundfile as sf

from . import pyAMT


MATLAB_AMT_PATH = "./matlab_toolboxes/amtoolbox"
MATLAB_AMT_MOD_PATH = "./matlab_toolboxes/TMST/"


class TestPyAMT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eng = matlab.engine.start_matlab()

        for path in [MATLAB_AMT_PATH, MATLAB_AMT_MOD_PATH]:
            if os.path.exists(path):
                cls.eng.eval(f"addpath(genpath('{path}'))", nargout=0)
            else:
                print(f"The path {path} does not exist.")

    @classmethod
    def tearDownClass(cls):
        cls.eng.quit()


    def test_auditory_filterbank(self):
        audio_path = './LaVoixHumaine_6s.wav'
        fmin, fmax = 70., 6700.

        sig, fs = sf.read(audio_path)
        sig = sig[:,0]
        py_gamma_responses, py_fc = pyAMT.auditory_filterbank(sig, fs, fmin, fmax)

        self.eng.eval(f"[sig, fs] = audioread('{audio_path}');", nargout=0)
        self.eng.eval(f"[response, fc] = auditoryfilterbank(sig(:,1), fs, {fmin}, {fmax});", nargout=0)
        mat_gamma_responses = np.squeeze(self.eng.workspace['response']).T
        mat_fc = np.squeeze(self.eng.workspace['fc'])

        np.testing.assert_allclose(mat_fc, py_fc,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_gamma_responses, py_gamma_responses,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_king2019_modfilterbank_updated(self):
        sig = np.random.rand(30, 10000)
        fs = 2500.
        mfmin, mfmax = 0.5, 200.
        modbank_Nmod = 200
        modbank_Qfactor = 1.

        py_outsig, py_mfc, py_step = pyAMT.king2019_modfilterbank_updated(sig, fs, mfmin, mfmax, modbank_Nmod, modbank_Qfactor)

        self.eng.workspace['sig'] = matlab.double(sig.tolist())
        self.eng.workspace['fs'] = fs
        matlab_code = f"""
            flags = struct('do_LP_150_Hz', 0, 'do_phase_insens_hilbert', 0);
            kv = struct('mflow', {mfmin}, 'mfhigh', {mfmax}, 'modbank_Nmod', {modbank_Nmod}, 'modbank_Qfactor', {modbank_Qfactor});
            [outsig, mfc, step] = king2019_modfilterbank_updated(sig, fs, 'argimport', flags, kv);
        """
        self.eng.eval(matlab_code, nargout=0)
        mat_outsig = np.transpose(self.eng.workspace['outsig'])
        mat_mfc = np.squeeze(self.eng.workspace['mfc'])
        mat_step = self.eng.workspace['step']

        np.testing.assert_allclose(mat_outsig, py_outsig,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_mfc, py_mfc,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.squeeze(mat_step['a']), py_step['a'],
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.squeeze(mat_step['b']), py_step['b'],
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unit tests for the pyAMT module.')
    parser.add_argument('--float_rel_tolerance', type=float, default=1.e-7,
                        help='Relative tolerance for floating point comparisons')
    parser.add_argument('--float_abs_tolerance', type=float, default=0,
                        help='Absolute tolerance for floating point comparisons')
    parser.add_argument('--verbosity', type=int, choices=[0,1,2], default=0,
                        help='Level of verbosity for test output')

    args = parser.parse_args()

    TestPyAMT.float_rel_tolerance = args.float_rel_tolerance
    TestPyAMT.float_abs_tolerance = args.float_abs_tolerance
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPyAMT)
    unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

