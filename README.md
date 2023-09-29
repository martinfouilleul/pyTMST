# pyTMST

Python port of the [MATLAB TMST toolbox](https://github.com/LeoVarnet/TMST) by
Léo Varnet. Currently a work in progress.

## Installation

1. Clone the repository: ```git clone https://github.com/anzic0/pyTMST```
2. Change into the local repository clone's directory: ```cd pyTMST```
3. Create a virtual environment (recommended): ```python3 -m venv venv```
4. Activate the virtual environment: ```source venv/bin/activate```
5. Install requirements: ```pip install -r requirements.txt```
6. Install package: ```pip install .```

## Testing

To run the test script, the following MATLAB toolboxes must be put in a directory
called ```matlab_toolboxes```:
- [TMST](https://github.com/LeoVarnet/TMST)
- [Auditory Modelling Toolbox](https://amtoolbox.org/)
- [YIN](http://audition.ens.fr/adc/sw/yin.zip)

Furthermore, the [MATLAB Engine API for Python](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) corresponding to your local MATLAB version needs to be installed.
