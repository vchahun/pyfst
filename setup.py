from distutils.core import setup
from distutils.extension import Extension

ext_modules = [
    Extension(name='fst._fst',
        sources=['fst/_fst.cpp'],
        libraries=['z', 'fst'],
        extra_compile_args=['-O2'])
]

setup(
    name='pyfst',
    version='1.0dev',
    url='http://github.com/vchahun/pyfst',
    author='Victor Chahuneau',
    author_email='?',
    packages=['fst'],
    ext_modules=ext_modules
)
