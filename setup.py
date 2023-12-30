from setuptools import setup


setup(
    name='pyTMST',
    version='1.0.0',
    packages=['pyTMST', 'pyTMST.pyAMT', 'pyTMST.pyLTFAT', 'pyTMST.utils', 'pyTMST.yin'],
    description='Python port of the MATLAB TMST toolbox',
    install_requires=[
        'numpy',
        'scipy',
        'soundfile',
        'gammatone',
        'librosa'
        ],
    author='Anton Zickler',
    author_email='anton.zickler@proton.me',
    package_data={
        'pyTMST': ['LICENSE.md'],
        'pyTMST': ['LICENSES/*'],
        },
)

