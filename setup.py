# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("blocky_score",
              sources=["blocky_score.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-O3"]) 
]

setup(
    ext_modules = cythonize(ext_modules)
)
