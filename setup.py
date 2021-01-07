from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("./requirements.txt", "r") as f:
    required = f.read().splitlines()

version_file = "rl_algorithms/version"


def get_version():
    version = open(version_file, "r", encoding="utf-8").read().strip()
    return version


setup(
    name="rl_algorithms",
    version=get_version(),
    author="medipixel",
    author_email="kh.kim@medipixel.io",
    description="Reinforcement Learning algorithms which are being used for research \
        activities at Medipixel.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/medipixel/rl_algorithms.git",
    keywords="reinforcement-learning python machine learning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=required,
    include_package_data=True,
    zip_safe=False,
)
