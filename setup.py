from setuptools import setup, find_packages

with open("README.MD", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "Named_Entity_Recognition "
AUTHOR_USER_NAME = "Akash soni"
SRC_REPO = "ner"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="Akash soni",
    description="A small package for NER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/akash-soni/Named_Entity_Recognition",
    author_email="akash.200287@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.6",
    install_requires=LIST_OF_REQUIREMENTS
)