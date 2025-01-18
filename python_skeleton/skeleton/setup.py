from setuptools import setup
from Cython.Build import cythonize

setup(
    name="BountyEquityModule",
    ext_modules=cythonize("bounty_equity.pyx"),  # or ["bounty_equity.pyx"]
)