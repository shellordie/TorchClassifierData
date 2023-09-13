from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = "Small pytorch utility to Import,Split,Normalize and Visualize custom dataset for classification tasks"

setup(
        name="TorchClassifierData", 
        version=VERSION,
        author="charles TCHANAKE",
        author_email="shellordie@gmail.com",
        url="https://github.com/shellordie/TorchClassifierData",
        description=DESCRIPTION,
        long_description_content_type="text/markdown",
        long_description=open("README.md","r",encoding="utf-8").read(),
        packages=find_packages(),
        install_requires=["numpy","opencv-python",
                          "matplotlib","pandas","torch"],  
        keywords=["Computer-vision","pytorch",
        "preprocessing","Images","Dataset","TorchClassifierData"],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
