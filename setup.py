import os, shutil
from setuptools import setup, find_packages

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(CUR_PATH, 'build')
if os.path.isdir(path):
    print('INFO del dir ', path)
    shutil.rmtree(path)


setup(
    name = 'queuetheory',
    author = 'Manning',
    author_email = 'manningcyrus@qq.com',
    version = '1.0.0',
    description = 'This package is used to conduct the queue model.',
    python_requires = '>=3.12',
    packages = find_packages(),
    include_package_data = True,
    exclude_package_data = {'docs':['1.txt']},
    install_requires = [
        'numpy>=2.0.0',
        'scipy>=1.15.0',
        'sympy>=1.13.0',
        'matplotlib>=3.10.0',
        'pytest>=8.3.0'
    ],

)