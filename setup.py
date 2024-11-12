#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='hubert',
      version='0.1.0',
      description='SLM.',
      author='Fatlonder Cakolli',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=["tqdm", "looseversion",],
      python_requires='>=3.8',
      include_package_data=True)