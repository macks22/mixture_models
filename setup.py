from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np


extensions = [
    Extension("ccomp", ["mixture/ccomp.pyx"],
        include_dirs=[np.get_include()]),
    Extension("cdist", ["mixture/cdist.pyx"],
        include_dirs=[np.get_include()])
]

setup(
    name="mixture_models",
    ext_modules=cythonize(extensions))
