import os
import sys
import sysconfig
from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension


def config_cython():
    sys_cflags = sysconfig.get_config_var('CFLAGS')

    # failure if import error
    from Cython.Build import cythonize
    ret = []
    # path = 'python/graph_rewriter/_cython'
    path = 'graph_rewriter/_cython'
    for fn in os.listdir(path):
        if fn.endswith('.pyx'):
            ret.append(Extension(
                'graph_rewriter.%s' % fn[:-4],
                sources=['%s/%s' % (path, fn)],
                include_dirs=["..", "../taso_ext/src", "/usr/local/cuda/include"],
                libraries=["taso_rl", "taso_runtime"],
                extra_compile_args=['-DUSE_CUDNN', '-std=c++11'],
                extra_link_args=[],
                language='c++',
            ))
    return cythonize(ret, compiler_directives={'language_level': 3})

setup(
    name='graph_rewriter',
    version='0.1.0',
    description='',
    zip_safe=False,
    url='na',
    ext_modules=config_cython(),
)
