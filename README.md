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
conda install -c conda-forge notebook ipykernel jupyterlab
pip install -e .
```

# Using midap-tools

The best way to get started is to follow the guide. 
It includes automatic download of example data that will allow you to tests midap-tools directly with some real data.
the guide is a collection of multiple .ipynb (jupyter notebook) files, one for each chapter (see [notebooks/guide/](notebooks/guide/))

the easiest way to run it on your system is to open it in jupyter lab.

```
jupyter lab
```

then navigate to notebooks/guide and open the first file

alternatively, you can use VSCode to run midap-tools. This is the recommended approach for power users that want to develop their own custom analysis methods.
VSCode may not detect the kernel in the jupyter notebooks the first time you open the project folder. To fix this issue, simply restart VSCode once, and it should work