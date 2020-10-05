#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

# The text of the README file
with open("README.md") as readme_file:
    README = readme_file.read()

setup(
    name="dexpression_pytorch",
    version="0.1.0",
    description="Pytorch implementation of DeXpression for facial emotion recognition.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rdgozum/dexpression-pytorch",
    author="Ryan Paul Gozum",
    author_email="ryanpaul.gozum@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["dexpression_pytorch"],
    packages=find_packages(),
    install_requires=[],
    package_data={},  # Optional
    data_files=[],  # Optional
    entry_points={},  # Optional
    project_urls={},  # Optional
)
