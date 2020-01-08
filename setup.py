import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# pylint: disable=line-too-long
setuptools.setup(
    name="rl_algorithms",
    version="0.0.1",
    author="medipixel",
    author_email="kh.kim@medipixel.io",
    description="Reinforcement Learning algorithms which are being used for research activities at Medipixel.",  # noqa
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/medipixel/rl_algorithms.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    zip_safe=False,
)
