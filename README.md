# ai-lecture-project

This project is the practical part of an Artificical Intelligence lecture at University of Augsburg.

This project uses Python 3.11 based on Anaconda (https://www.anaconda.com/distribution/).

## Getting started

The file 'requirements.txt' lists the required packages.

1. We recommend to use a virtual environment to ensure consistency, e.g.   
`conda env create -f environment.yml`

2. To later update your dependencies:  
`conda env update -f environment.yml` 

3. Install Black Formatter and Flake8 linter. In VSCode, both are available as official Microsoft extensions. Do always format your code with black before commiting your code.

## Software Tests
This project contains some software tests based on Python Unittest (https://docs.python.org/3/library/unittest.html). 
Run `python -m unittest` from the command line in the repository root folder to execute the tests. This should automatically search all unittest files in modules or packages in the current folder and its subfolders that are named `test_*`.

## Final evalution
We've handed in `evaluation.py` with about the same logic as in the code skeleton we received at the beginning. We hope that makes it east to run the final tests. As a final model, we chose the model resulting from the rule-based approach.

## Approach documentation
[Check out this file.](./approaches.md)