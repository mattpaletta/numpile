import inspect
import os
import subprocess
import sys
from distutils.command.build import build
from setuptools.command.install import install

try:
    from setuptools import setup, find_packages
except ImportError:
    import ez_setup

    ez_setup.use_setuptools()
    from setuptools import setup, find_packages

setup(
    name="numpile",
    version="0.0.1",
    url='https://github.com/mattpaletta/numpile',
    packages=find_packages(),
    include_package_data=True,
    install_requires=["llvmlite", "numpy"],
    setup_requires=[],
    author="Matthew Paletta",
    author_email="mattpaletta@gmail.com",
    description="Numpile tutorial",
    license="BSD",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Communications',
    ],
)
