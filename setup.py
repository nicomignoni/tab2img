import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tab2img",
    version="0.9.0",
    author="Nicola Mignoni",
    author_email="",
    description="A tool to convert tabular data into images, in order to be used by CNN. Inspired by the 'DeepInsight' paper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicomignoni/tab2img",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
