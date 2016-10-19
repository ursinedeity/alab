#!/usr/bin/env python
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

install_requires = [
    'numpy>=1.9', 
    'scipy>=0.16', 
    'pandas>=0.17',
    'h5py>=2.5', 
    'matplotlib>=1.5',
    'seaborn',
    'cython',
    'imp>=2.4',
]
tests_require = [
    'mock'
]


extras_require = {
    'docs': [
        'Sphinx>=1.1', 
    ]
}
setup(
        name = 'alab', 
        version = '1.0.0', 
        author = 'Nan Hua', 
        author_email = 'nhua@usc.edu', 
        url = 'https://github.com/ursinedeity/alab', 
        description = '3D Modeling Pipeline Workflows',
        packages=['alab'],
        package_data={'alab' : ['genomes/*']},
        install_requires=install_requires,
        tests_require=tests_require,
        extras_require=extras_require,
        ext_modules=cythonize("alab/numutils.pyx"),
        include_dirs=[numpy.get_include()]
)
