from setuptools import setup, find_packages
import pathlib

import pkg_resources
import setuptools

#with pathlib.Path('requirements2.txt').open() as requirements_txt:
#    install_requires = [
#        str(requirement)
#        for requirement
#        in pkg_resources.parse_requirements(requirements_txt)
#    ]
    
    
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='alphatracker2',
      version='2.0.0',
      description='AlphaTracker development testing script',
      author='Aneesh Bal',
      author_email='aneesh.s.bal@gmail.com',
      packages=find_packages(),
      #install_requires=required,
      package_data={'': ['*.yaml']},
      setup_requires=['setuptools_scm'],
      include_package_data=True,
     )
