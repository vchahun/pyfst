import sys
import os
from distutils.core import setup
from distutils.extension import Extension

INC, LIB = [], []

# MacPorts
if sys.platform == 'darwin' and os.path.isdir('/opt/local/lib'):
    INC.append('/opt/local/include')
    LIB.append('/opt/local/lib')

ext_modules = [
    Extension(name='fst._fst',
        sources=['fst/_fst.cpp'],
        libraries=['z', 'fst'],
        extra_compile_args=['-O2'],
        include_dirs=INC,
        library_dirs=LIB)
]

long_description = """
pyfst
=====

A Cython wrapper of the OpenFst_ library.

Requires OpenFst 1.3

.. _OpenFst: http://www.openfst.org

Example usage::

    from fst import SimpleFst

    t = SimpleFst()

    t.add_arc(0, 1, 'a', 'A', 0.5)
    t.add_arc(0, 1, 'b', 'B', 1.5)
    t.add_arc(1, 2, 'c', 'C', 2.5)

    t[2].final = 3.5

    t.shortest_path() # 2 -(a:A/0.5)-> 1 -(c:C/2.5)-> 0/3.5 

"""

setup(
    name='pyfst',
    version='0.1.3',
    url='http://github.com/vchahun/pyfst',
    author='Victor Chahuneau',
    description='Cython wrapper of the OpenFst library.',
    long_description=long_description,
    classifiers=['Topic :: Text Processing :: Linguistic'],
    packages=['fst'],
    ext_modules=ext_modules
)
