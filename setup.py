from setuptools import setup, find_packages

setup(
    name="cluster",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0'
    ],
    python_requires='>=3.8',
)