import pathlib
from setuptools import setup, find_packages

setup(
    name="imagedataset",
    version="0.1",
    packages=find_packages(),
    install_requires=pathlib.Path("requirements.txt").read_text().splitlines(),,
)
