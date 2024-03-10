#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Neural Networks Compression",
    author="Lukasz Staniszewski",
    author_email="lukaszstaniszewski10@gmail.com",
    url="https://github.com/lukasz-staniszewski/neural-networks-compression",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
