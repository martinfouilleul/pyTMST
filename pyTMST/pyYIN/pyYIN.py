import matlab.engine
import numpy as np


def mock_yin(sig, fs, f0_min, f0_max, thresh, hop):
    eng = matlab.engine.start_matlab()
    eng.eval(f"addpath(genpath('./matlab_toolboxes/yin'))", nargout=0)
    eng.eval(f"P = struct('sr', {fs}, 'minf0', {f0_min}, 'maxf0', {f0_max}, 'thresh', {thresh}, 'hop', {hop});", nargout=0)
    eng.workspace['sig'] = matlab.double(sig.reshape(-1,1).tolist())
    eng.workspace['fs'] = matlab.double(fs)
    eng.eval(f"r = yin(sig, P);", nargout=0)
    f0 = np.squeeze(eng.workspace['r']['f0'])
    ap0 = np.squeeze(eng.workspace['r']['ap0'])
    return f0, ap0
    
