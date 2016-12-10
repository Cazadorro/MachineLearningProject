from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

cython_dir = "cythonsrc/"
cppsrc_dir = "cythonsrc/cppsource/"


source2 = [cython_dir + "floyedwarshalls.pyx", ]

compile_args = ["-std=c++11", "-g"]
language = 'c++'
link_args = ["-std=c++11"]

extensions = [
    Extension("floyedwarshalls", source2,
              language=language, gdb_debug=True,
              extra_compile_args=compile_args,
              extra_link_args=link_args)
]

setup(
    name="fwapsp",
    ext_modules=cythonize(extensions)
)
