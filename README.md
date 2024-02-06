# ai-lecture-project

This project is written with Python `3.8` based on Anaconda (https://www.anaconda.com/distribution/).
(The project can be upgraded to a higher Python version if required. )

## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.   
`conda create -n ai-project python=3.8`

2. Install the dependencies:  
`conda install -c conda-forge --file requirements.txt` 


## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html). 
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This should automatically search all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.