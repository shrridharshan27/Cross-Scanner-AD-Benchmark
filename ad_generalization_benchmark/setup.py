from setuptools import setup, find_packages
from pathlib import Path

# Map package root so `import ad_generalization_benchmark...` resolves properly
setup(
    name="ad_generalization_benchmark",
    version="0.1.0",
    package_dir={"": ".."},
    packages=["ad_generalization_benchmark", *(f"ad_generalization_benchmark.{p}" for p in find_packages())],
    python_requires=">=3.10",
)
