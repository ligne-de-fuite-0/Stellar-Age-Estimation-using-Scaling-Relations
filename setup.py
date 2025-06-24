from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stellar-age-estimation",
    version="1.0.0",
    author="ligne-de-fuite-0",
    author_email="your.email@example.com",
    description="A Python package for estimating stellar ages using scaling relations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ligne-de-fuite-0/stellar-age-estimation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8", "sphinx"],
        "notebooks": ["jupyter", "notebook"],
    },
    include_package_data=True,
    package_data={
        "stellar_age": ["data/*.csv", "data/*.py"],
    },
)