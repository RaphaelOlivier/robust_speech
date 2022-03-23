#!/usr/bin/env python3
import os
import sys
import site
import setuptools
from distutils.core import setup


# Editable install in user site directory can be allowed with this hack:
# https://github.com/pypa/pip/issues/7953.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md") as f:
    long_description = f.read()

with open(os.path.join("robust_speech", "version.txt")) as f:
    version = f.read().strip()

setup(
    name="robust_speech",
    version=version,
    description="Adversarially Robust Speech Recognition toolkit built over SpeechBrain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Raphael Olivier",
    author_email="rolivier@cs.cmu.edu",
    packages=setuptools.find_packages(),
    package_data={"robust_speech": ["version.txt", "log-config.yaml"]},
    install_requires=[
        "torch>=1.7,<=1.11",
        "speechbrain==0.5.11",
        "transformers>=4.15.0",
        "torchvision",
        "advertorch @ git+https://github.com/BorealisAI/advertorch.git"
    ],
    python_requires=">=3.7",
    url="https://github.com/RaphaelOlivier/robust_speech",
)
