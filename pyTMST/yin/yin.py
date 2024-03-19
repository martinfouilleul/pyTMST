import numpy as np
from .librosa_yin_ap import yin_ap


def mock_yin(sig, fs, f0_min, f0_max, thresh, hop):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.eval(f"addpath(genpath('./matlab_toolboxes/yin'))", nargout=0)
    eng.eval(f"P = struct('sr', {fs}, 'minf0', {f0_min}, 'maxf0', {f0_max}, 'thresh', {thresh}, 'hop', {hop});", nargout=0)
    eng.workspace['sig'] = matlab.double(sig.reshape(-1,1).tolist())
    eng.workspace['fs'] = matlab.double(fs)
    eng.eval(f"r = yin(sig, P);", nargout=0)
    f0 = 440. * 2 ** np.squeeze(eng.workspace['r']['f0'])
    ap0 = np.squeeze(eng.workspace['r']['ap0'])
    return f0, ap0


def librosa_yin(sig, fs, f0_min, f0_max, thresh, hop):
    f0, ap0 = yin_ap(sig, sr=fs,
                     fmin=f0_min, fmax=f0_max,
                     trough_threshold=thresh,
                     hop_length=hop,
                     win_length=-(fs // -f0_min), # ceiling division
                     frame_length=10000
                     )
    return f0, ap0
