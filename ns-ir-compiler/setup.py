from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = [
    line.strip()
    for line in Path("requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#") and not line.startswith("llvmlite") and not line.startswith("islpy") and not line.startswith("pybind11")
]

setup(
    name="nsir-compiler",
    version="0.1.0",
    description="Neural-Symbolic IR Compiler — Learned cost models for polyhedral compilers",
    author="NS-IR Team",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nsir=cli.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
)
