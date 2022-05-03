from setuptools import setup, find_packages
import trading_machine_learning_models

classifiers = [
    'Developement Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9.5'
]

setup(
name = 'trading_machine_learning_models',
version = trading_machine_learning_models.__version__,
description = 'trading_machine_learning_models modules is a Python package that can run multiple pre-determined machine learning models. The package allows you to set a multiple set of values for hyper-parameter tuning.',
url= 'https://github.com/Iankfc/trading_machine_learning_models',
author='ece',
author_email='odesk5@outlook.com',
license = 'None',
classifiers=classifiers,
keywords='None',
packages=find_packages(),
use_scm_version=True,
include_package_data=True,
setup_requires=['setuptools_scm'],
install_requires = open('requirements.txt','r').read().split('\n')[:-1]
)