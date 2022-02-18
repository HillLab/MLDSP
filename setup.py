"""
Setup file for MLDSP_core
"""
from pathlib import Path

from setuptools import setup, find_packages

from MLDSP_core.__version__ import version

parent = Path(__file__).parent.resolve()
readme = parent.joinpath('README.md')
reqs = parent.joinpath('requirements.txt')

with open(readme, encoding='utf-8') as readm, open(reqs) as requ:
    long_description = readm.read()
    requirements = requ.read().strip().split()

setup(
    name='MLDSP',
    version=version,
    packages=find_packages(),
    # package_dir={"": "MLDSP"},
    url='https://github.com/HillLab/MLDSP',
    license='GNU',
    author='Daniel Olteanu, Jose Sergio Hleap',
    # ADD other authors if required
    author_email='dolteanu@uwo.ca, jshleap@sharcnet.ca',
    description='Machine Learning with Digital Signal Processing',
    python_requires='>=3.7',
    install_requires=[requirements],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
