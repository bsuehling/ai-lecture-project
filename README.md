# ai-lecture-project

This project uses Python 3.11 based on Anaconda (https://www.anaconda.com/distribution/).

## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.   
`conda env create -f environment.yml`

2. To later update your dependencies:  
`conda env update -f environment.yml` 

3. Install Black Formatter and Flake8 linter. In VSCode, both are available as official Microsoft extension. Do always format your code with black before commiting your code.

## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html). 
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This should automatically search all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.