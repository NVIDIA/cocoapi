from os.path import abspath, join, exists
from setuptools import setup, Extension
from sys import version_info
import os

# Provide cythonize fallback: if Cython is available build from .pyx,
# otherwise use the checked-in .c/.cpp files.
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except Exception:
    cythonize = None
    USE_CYTHON = False

# Ensure numpy is available at build time (pyproject.toml should include numpy in build-system.requires)
import numpy as np

# Workaround for Python 3.12 on this repo (keeps compatibility with existing behavior)
if (version_info.major, version_info.minor) >= (3, 12) and not exists("pycocotools/_mask.c"):
    open("pycocotools/_mask.c", "w").close()

mask_source = 'pycocotools/_mask.pyx' if USE_CYTHON else 'pycocotools/_mask.c'
ext_source_maskapi = abspath(join('..', 'common', 'maskApi.c'))

ext_modules = [
    Extension(
        name='pycocotools._mask',
        sources=[ext_source_maskapi, mask_source],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        include_dirs=[np.get_include(), abspath(join('..', 'common'))],
    ),
    Extension(
        name='ext',
        sources=['pycocotools/ext.cpp', 'pycocotools/simdjson.cpp'],
        extra_compile_args=['-O3', '-Wall', '-shared', '-fopenmp', '-std=c++17', '-fPIC'],
        include_dirs=[np.get_include(), 'pycocotools'],
        library_dirs=[abspath(join(np.get_include(), '..', 'lib'))],
        libraries=['npymath', 'gomp'],
        language='c++',
    )
]

# If Cython is present, cythonize the extensions so .pyx is compiled
if USE_CYTHON:
    ext_modules = cythonize(ext_modules, language_level=3)

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir={'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'matplotlib>=2.1.0',
    ],
    version='2.0+nv0.8.1',
    ext_modules=ext_modules,
)
