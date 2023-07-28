# Glocal Explanations of Expected Goal Models in Football

The repository contains the codes to reproduce the results presetnted in the paper.

The main code for aSHAP values was written in the `Rcodes.Rmd` file. Moreover, the folder `changes_forester` contains additional required files which introduce some changes into the forester package.

The aggregated profiles codes were written in Python and are available in Jupyter notebooks.

## Run the codes locally


1. To install `R` and `Python`, you have to run a script from the project's main directory:
```console
./create_environment_scripts/install_R_Python
```
2. To install all the necessary libraries and to create `Python` virtual environment, you have to run a script from the project's main directory:
```console
./create_environment_scripts/create_environment
```
3. To open `jupyterlab`, run in a command line from the project's main directory:
```console
source ./virtualenv/bin/activate

jupyter lab
```
