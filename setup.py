from distutils.core import setup
from distutils.extension import Extension

ext_modules = [
    Extension(name='fst',
        sources=['fst.cpp'],
        include_dirs=['.'],
        libraries=['z', 'fst', 'fstscript'],
        extra_compile_args=['-O2'])
]

setup(
    name='pyfst',
    ext_modules=ext_modules
)
