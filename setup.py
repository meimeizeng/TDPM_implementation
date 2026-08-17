from setuptools import find_packages, setup

setup(
    name="tdpm",
    version="1.0.0",
    description="Truncated Diffusion Probabilistic Models for image restoration",
    packages=find_packages(include=["tdpm", "tdpm.*"]),
    python_requires=">=3.9",
)
