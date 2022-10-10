from setuptools import setup, Extension
import numpy as np
from numpy.distutils.misc_util import get_info

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

library_dirs = []
library_dirs += get_info('npymath')['library_dirs']

ext_modules = [
    Extension(
        name='pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        include_dirs = [np.get_include(), '../common'],
    ),
    Extension(
        name='ext',
        sources=['pycocotools/ext.cpp', 'pycocotools/simdjson.cpp'],
        extra_compile_args=['-O3', '-Wall', '-shared', '-fopenmp', '-std=c++17', '-fPIC'],
        include_dirs = [np.get_include(), 'pycocotools'],
        library_dirs=library_dirs,
        libraries=['npymath', 'gomp']
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0',
    ],
    version='2.0+nv0.7.0',
    ext_modules= ext_modules
)
