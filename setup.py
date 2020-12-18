import glob
import os
from pathlib import Path
from setuptools import setup

version = '0.0.1'

# copy over packages
directories = ['src']

pkg_dirs = []


def recursive_directories(dirs):
    for directory in dirs:
        pkg_dirs.append(directory)
        files = glob.glob(directory+'/*')
        for f in files:
            if os.path.isdir(f):
                recursive_directories((f,))


recursive_directories(directories)


setup(name='gf_opt',
      version=version,
      url='https://github.com/pjstanle/gf_opt',
      description='A collection of gradient-free optimizers',
      author='PJ Stanley',
      author_email='pj.stanley@nrel.gov',
      packages=pkg_dirs,
      )
