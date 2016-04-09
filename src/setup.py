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
    name="ipmlr",
    ext_modules=cythonize(extensions))

# setup(
#     name="ipmlr",
#     ext_modules=cythonize(
#         "mixture/*.pyx",
#         include_dirs=[np.get_include()]
#     )
# )
