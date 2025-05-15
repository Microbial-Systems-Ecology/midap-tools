# MIDAP-Tools

MIDAP-tools is a flexible and user-friendly software for the post processing of data generated with [midap](https://github.com/Microbial-Systems-Ecology/midap). Tools enable loading and mutation of entire experiments, visualization of raw data and processed data in a number of ways and a suite of analyses such as growth rate, local neighborhood and time based analyses.

# Installation

to install midap-tools follow these steps

move to a folder where you want to download the package from github.
download the package with git clone, create a conda environment and install from the setup.py following the steps outlined below

```
git clone https://github.com/Microbial-Systems-Ecology/midap-tools.git
cd midap-tools
conda create --name midap-tools python=3.10
conda activate midap-tools
pip install -e .
```