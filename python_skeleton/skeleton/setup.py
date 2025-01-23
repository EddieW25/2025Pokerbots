from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
       "bounty_equity.pyx",
       include_path=["C:/Users/game/anaconda3/envs/Pokerbots/Lib/site-packages/eval7"]
    )
)

# run python setup.py build_ext --inplace