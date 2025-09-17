"""Setup script for POS Tagger project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="unsupervised-pos-vqvae-hindi",
    version="0.1.0",
    author="Chini Lahoti",
    author_email="chini.s.lahoti@gmail.com",
    description="Unsupervised Part-of-Speech Induction for a Morphologically Rich Language using Gumbel Softmax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chinilahoti/unsupervised-pos-vqvae-hindi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "pos-tagger=pos_tagger.main:main",
        ],
    },
)