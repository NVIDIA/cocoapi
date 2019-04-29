from setuptools import setup, Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    ),
    Extension(
        'ext',
        sources=['pycocotools/ext.cpp'],
        extra_compile_args=['-O3', '-Wall', '-shared', '-fopenmp', '-std=c++11', '-fPIC'],
        extra_link_args=['-lgomp'],
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
        'pybind11>=2.2',
    ],
    version='2.0+nv0.2.1',
    ext_modules= ext_modules
)
