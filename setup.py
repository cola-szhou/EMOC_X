import distutils.command.build as _build
import os
import sys
from distutils import spawn
from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages


__version__ = "0.0.1-alpha.1"


setup(
    name="emoc",
    version=__version__,
    packages=find_packages(),
    description="EMOC+X: Evolutionary Multi-objective Optimization in C++",
    long_description=open("README.md").read(),
    author="",
    author_email="",
    # license="",
    url="https://colalab.ai/EMOCDoc/",
    install_requires=[
        "numpy",
        "GPy",
        "GPyOpt",
        "Plotly",
        "matplotlib",
        "pandas",
        "emoc-cpp",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        # 'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    include_package_data=True,
    package_data={
        "emoc": ["pf_data/*/*.pf"],
    },
)
