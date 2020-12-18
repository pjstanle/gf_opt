import glob
import os
from pathlib import Path
from setuptools import setup

version = '0.0.1'

setup(name='gfopt',
      version=version,
      url='https://github.com/pjstanle/gfopt',
      description='A collection of gradient-free optimizers',
      author='PJ Stanley',
      author_email='pj.stanley@nrel.gov',
      packages=["gfopt"],
      install_requires=[
        "numpy >= 1.6",
        ]
      )
