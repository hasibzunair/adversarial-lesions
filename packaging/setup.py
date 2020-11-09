import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="melanet",
    version="0.0.2",
    author="Hasib Zunair",
    author_email="hasibzunair@gmail.com",
    description="MelaNet demo model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasibzunair/adversarial-lesions",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)