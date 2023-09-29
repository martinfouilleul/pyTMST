import unittest
import os
import argparse

import matlab.engine
import numpy as np
import soundfile as sf
import scipy.signal
import yin

import pyTMST

MATLAB_TOOLBOX_PATHS = {
    'TMST': './matlab_toolboxes/TMST',
    'Auditory Modeling': './matlab_toolboxes/amtoolbox',
    'YIN': './matlab_toolboxes/yin'
}


class TestTMST(unittest.TestCase):
    float_rel_tolerance = 1.e-7
    float_abs_tolerance = 0

    @classmethod
    def setUpClass(cls):
        cls.eng = matlab.engine.start_matlab()

        for path in MATLAB_TOOLBOX_PATHS.values():
            if os.path.exists(path):
                cls.eng.eval(f"addpath(genpath('{path}'))", nargout=0)
            else:
                print(f"The path {path} does not exist.")


    @classmethod
    def tearDownClass(cls):
        cls.eng.quit()


    def test_hilbert(self):
        t = np.arange(0, 1 + 1/1024, 1/1024)
        f = 60
        x = np.sin(2 * np.pi * f * t)
        matlab_result = np.squeeze(self.eng.hilbert(x))
        python_result = scipy.signal.hilbert(x)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_freq_to_erb(self):
        freq = 70.;
        matlab_result = self.eng.freqtoaud(freq)
        python_result = pyTMST.freq_to_erb(freq)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)


    def test_erb_to_freq(self):
        erb = 31.6
        matlab_result = self.eng.audtofreq(erb)
        python_result = pyTMST.erb_to_freq(erb)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

    def test_erb_filt_bw(self):
        fc = 70.
        mat_bw = self.eng.audfiltbw(fc)
        py_bw = pyTMST.erb_filt_bw(fc)
        np.testing.assert_allclose(mat_bw, py_bw,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

    def test_erbspace_bw(self):
        fmin, fmax = 70., 6700.
        matlab_result = np.array(self.eng.erbspacebw(fmin, fmax)).squeeze()
        python_result = pyTMST.erbspace_bw(fmin, fmax, bw=1)
        np.testing.assert_allclose(matlab_result, python_result,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

    def test_apply_gammatone_filterbank_mono(self):
        audio_path = './LaVoixHumaine_6s.wav'
        fmin, fmax = 70., 6700.

        sig, fs = sf.read(audio_path)
        sig = sig[:,0]
        py_gamma_responses, py_fc = pyTMST.apply_gammatone_filterbank(sig, fs, fmin, fmax)

        self.eng.eval(f"[sig, fs] = audioread('{audio_path}');", nargout=0)
        self.eng.eval(f"[response, fc] = auditoryfilterbank(sig(:,1), fs, {fmin}, {fmax});", nargout=0)
        mat_gamma_responses = np.squeeze(self.eng.workspace['response']).T
        mat_fc = np.squeeze(self.eng.workspace['fc'])

        np.testing.assert_allclose(mat_fc, py_fc,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_gamma_responses, py_gamma_responses,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

    def test_define_modulation_axis(self):
        mfmin, mfmax = 0.5, 200
        nf = 200

        py_f_spectra, py_f_spectra_intervals = pyTMST.define_modulation_axis(mfmin, mfmax, nf)

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

        py_pxx = pyTMST.periodogram(sig, fs, freqs)

        self.eng.eval(f"[pxx, f] = periodogram({matlab.double(sig.tolist())},[],{freqs},{fs},'psd');", nargout=0)
        mat_pxx = np.squeeze(self.eng.workspace['pxx'])

        np.testing.assert_allclose(mat_pxx, py_pxx,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)

    def test_AMa_spectrum_mono(self):
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
        np.testing.assert_allclose(np.squeeze(mat_step.t), py_step.t,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.squeeze(mat_step.f_bw), py_step.f_bw,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.transpose(mat_step.gamma_responses), py_step.gamma_responses,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.transpose(mat_step.E), py_step.E,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.squeeze(mat_step.mf), py_step.mf,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(np.squeeze(mat_step.mfb), py_step.mfb,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)
        np.testing.assert_allclose(mat_AMa_spec, py_AMa_spec,
                                   rtol=self.float_rel_tolerance, atol=self.float_abs_tolerance)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests specified in ./test_specs.json for pyTMST module.')
    parser.add_argument('--float_rel_tolerance', type=float, default=1.e-7,
                        help='Relative tolerance for floating point comparisons')
    parser.add_argument('--float_abs_tolerance', type=float, default=0,
                        help='Absolute tolerance for floating point comparisons')
    parser.add_argument('--verbosity', type=int, choices=[0,1,2], default=0,
                        help='Level of verbosity for test output')

    args = parser.parse_args()

    TestTMST.float_rel_tolerance = args.float_rel_tolerance
    TestTMST.float_abs_tolerance = args.float_abs_tolerance
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTMST)
    unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

