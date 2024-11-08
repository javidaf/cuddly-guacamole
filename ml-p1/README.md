
## Project 1 - FYS-STK4155

You can explore the results from the project by exploring the notebooks in the `Notebooks` folder. The notebooks are named according to the task they are solving.

if you want to run the final part of the project, that is comparing the different regression methods with bootstrap and k-fold cross validation on the SRTM data, you can run the following command in the terminal:


## Setup with conda environment
1. download the environment.yml and create a conda environment using the environment.yml file. 
2. activate the environment by running the following command in the terminal:
```bash
conda activate p1
```

download and install the project package by running the following command in the terminal:

```bash	
pip install git+https://github.com/javidaf/cuddly-guacamole.git
```

You can then run the following command in the terminal to run the final part of the project:

```bash
python __main__.py --path ml-p1\DataFiles\SRTM_data_Norway_1.tif --np 60 --degree 12 --scaling standard --lambda_ridge 0.1 --alpha_lasso 0.00158
```


