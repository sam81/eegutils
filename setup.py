#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.distutils.core import setup
#from numpy.distutils.core import Extension
#ext1 = Extension(name = 'libforbdf',
#                 sources = ['libforbdf.pyf','libforbdf.f95'])
setup(name="eegutils",    
    version="0.0.4",
      py_modules=["eegutils"],
      author="Samuele Carcagno",
      author_email="sam.carcagno@gmail.com;",
      description="eegutils is a python library for processing electroencephalographic recordings.",
      long_description=\
      """
      eegutils is a python library for processing
      electroencephalographic recordings to extract
      event related potentials.
      The software is currently in **ALPHA** status.
      It is used internally in my lab and for the moment
      I don't have enough time to polish it. 
      """,
      license="GPL v3",
      url="none",
      requires=['numpy (>=1.6.1)', 'scipy (>=0.10.1)'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
          ]
      )
