import unittest
import os
import argparse

import matlab.engine
import numpy as np
import soundfile as sf

import pyTMST


MATLAB_TOOLBOX_PATHS = ["./matlab_toolboxes/TMST/", "./matlab_toolboxes/amtoolbox"]


class TestPyTMST(unittest.TestCase):
    float_rel_tolerance = 1.e-7
    float_abs_tolerance = 0

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


    def test_AMa_spectrum(self):
        audio_path = './LaVoixHumaine_6s.wav'
        mfmin, mfmax = 0.5, 200.
        modbank_Nmod = 200
        fmin, fmax = 70., 6700.

        sig, fs = sf.read(audio_path)
        sig = sig[:,0]
        py_AMa_spec, py_fc, py_mf, py_step = pyTMST.AMa_spectrum(sig, fs, mfmin, mfmax, modbank_Nmod, fmin, fmax)

        self.eng.eval(f"[sig, fs] = audioread('{audio_path}');", nargout=0)
        self.eng.eval(f"[AMspec, fc, mf, step] = AMspectrum(sig(:,1), fs);", nargout=0)
        mat_AMa_spec = np.squeeze(self.eng.workspace['AMspec'])
        mat_fc = np.squeeze(self.eng.workspace['fc'])
        mat_mf = np.squeeze(self.eng.workspace['mf'])
        mat_step = pyTMST.AMa_spec_params(**self.eng.workspace['step'])

        np.testing.assert_allclose(mat_fc, py_fc,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_mf, py_mf,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_AMa_spec, py_AMa_spec,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_AMi_spectrum(self):
        audio_path = './LaVoixHumaine_6s.wav'
        mfmin, mfmax = 0.5, 200.
        modbank_Nmod = 200
        modbank_Qfactor = 1
        fmin, fmax = 70., 6700.

        sig, fs = sf.read(audio_path)
        sig = sig[:,0]
        py_AMi_spec, py_fc, py_mf, py_step = pyTMST.AMi_spectrum(sig, fs, mfmin, mfmax, modbank_Nmod, modbank_Qfactor, fmin, fmax)
        
        self.eng.eval(f"[sig, fs] = audioread('{audio_path}');", nargout=0)
        self.eng.eval(f"[AMIspec, fc, mf, step] = AMIspectrum(sig(:,1), fs);", nargout=0)
        mat_AMi_spec = np.squeeze(self.eng.workspace['AMIspec'])
        mat_fc = np.squeeze(self.eng.workspace['fc'])
        mat_mf = np.squeeze(self.eng.workspace['mf'])
        mat_step = pyTMST.AMi_spec_params(**self.eng.workspace['step'])

        np.testing.assert_allclose(mat_fc, py_fc,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_mf, py_mf,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.transpose(mat_AMi_spec), py_AMi_spec,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unit tests for the pyTMST module.')
    parser.add_argument('--float_rel_tolerance', type=float, default=1.e-7,
                        help='Relative tolerance for floating point comparisons')
    parser.add_argument('--float_abs_tolerance', type=float, default=0,
                        help='Absolute tolerance for floating point comparisons')
    parser.add_argument('--verbosity', type=int, choices=[0,1,2], default=0,
                        help='Level of verbosity for test output')

    args = parser.parse_args()

    TestPyTMST.float_rel_tolerance = args.float_rel_tolerance
    TestPyTMST.float_abs_tolerance = args.float_abs_tolerance
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPyTMST)
    unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

