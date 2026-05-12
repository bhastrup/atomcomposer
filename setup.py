import os

from setuptools import setup, find_packages


def readme() -> str:
    with open("README.md") as f:
        return f.read()


version_dict = {}
with open(os.path.join("src", "version.py")) as file:
    exec(file.read(), version_dict)

setup(
    name="src",
    version=version_dict["__version__"],
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "gym",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "ase",
    ],
    zip_safe=False,
    test_suite="pytest",
    tests_require=["pytest"],
)
