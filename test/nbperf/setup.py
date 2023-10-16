from setuptools import setup, Extension
import numpy as np

module1 = Extension('mandel',
                    sources = ['pure_c_double.c'],
                    include_dirs = [np.get_include()] )

setup (name = 'mandel',
       version = '1.0',
       description = 'A Mandelbrot function in C using doubles',
       ext_modules = [module1] )
