"""Setup script for the package."""

import setuptools

VERSION = "0.0.4"

with open("README.md", "r") as file_object:
    LONG_DESCRIPTION = file_object.read()

setuptools.setup(
    name="sidlingvo",
    version=VERSION,
    author="Quan Wang",
    author_email="quanw@google.com",
    description="Lingvo utils for Google SVL team",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/google/speaker-id/tree/master/lingvo",
    packages=["."],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
