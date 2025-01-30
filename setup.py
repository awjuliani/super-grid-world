#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    "numpy",
    "gym",
    "matplotlib",
    "scipy",
    "opencv-python",
    "PyOpenGL",
    "PyVirtualDisplay",
    "torch",
    "glfw",
]

setup(
    name="sgw",
    version="0.1.0",
    description="Super Grid World",
    license="Apache License 2.0",
    author="Arthur Juliani",
    author_email="awjuliani@gmail.com",
    url="https://github.com/awjuliani/super-grid-world",
    packages=find_packages(),
    install_requires=required,
    include_package_data=True,
    package_data={"sgw": ["textures/*.png"]},
)
