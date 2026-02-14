"""
Setup script for pytorch-job-manager package.
This file is kept for backward compatibility.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pytorch-job-manager",
    version="0.1.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for managing PyTorch distributed training jobs on Kubeflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pytorch-job-manager",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gradio>=3.0.0",
        "kubernetes>=20.0.0",
        "kubeflow-training>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)
