from setuptools import setup



setup(
    name='pyTMST',
    py_modules=['pyTMST'],
    version='0.0.1',
    description='Python port of the MATLAB TMST toolbox',
    install_requires=[
        'numpy',
        'scipy',
        'soundfile',
        'gammatone',
        'yin'
        ],
    author='Anton Zickler',
    author_email='anton.zickler@gmail.com',
)

