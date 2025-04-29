<<<<<<< HEAD
from setuptools import setup, find_packages

setup(
    name='cuQSL',
    version='1.0.0',
    packages=find_packages(where="src"),  # 自动查找 src 下的包
    package_dir={"": "src"},              # 包根目录映射到 src
    include_package_data=True,
    install_requires=[
        'numpy>=1.18.5',
        'torch>=1.10.0',
        'matplotlib>=3.2.2',
        'pyevtk>=1.1.0',
    ],
    author='Chen Guoyin',
    author_email='gychen@smail.nju.edu.cn',
    description='Compute the QSL for a given point data based on CUDA',
=======
from setuptools import setup, find_packages

setup(
    name='cuQSL',
    version='1.0.0',
    packages=find_packages(where="src"),  # 自动查找 src 下的包
    package_dir={"": "src"},              # 包根目录映射到 src
    include_package_data=True,
    install_requires=[
        'numpy>=1.18.5',
        'torch>=1.10.0',
        'matplotlib>=3.2.2',
    ],
    author='Chen Guoyin',
    author_email='gychen@smail.nju.edu.cn',
    description='Compute the QSL for a given point data based on CUDA',
>>>>>>> d905b53dad074c2669079616db069ce6e0301523
)