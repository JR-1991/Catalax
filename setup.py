import setuptools
from setuptools import setup

setup(
    name="Catalax",
    version="0.0.0",
    author="Range, Jan",
    author_email="jan.range@simtech.uni-stuttgart.de",
    license="MIT License",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "pydantic==1.10.7",
        "dotted-dict==1.1.3",
        "sympy==1.11.1",
        "diffrax>=0.3.1",
        "sympy2jax>=0.0.4",
        "pandas",
        "tqdm",
        "numpy",
        "numpyro",
        "arviz",
        "corner",
        "optax",
        "equinox",
    ],
)
