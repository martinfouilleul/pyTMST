from setuptools import setup


setup(
    name='pyTMST',
    packages=['pyTMST', 'pyTMST.pyAMT', 'pyTMST.pyLTFAT', 'pyTMST.utils'],
    version='0.1.0',
    description='Python port of the MATLAB TMST toolbox',
    install_requires=[
        'numpy',
        'scipy',
        'soundfile',
        'gammatone',
        ],
    author='Anton Zickler',
    author_email='anton.zickler@proton.me',
    package_data={
        'pyTMST': ['LICENSE.md'],
        'pyTMST': ['LICENSES/*'],
        },
)

