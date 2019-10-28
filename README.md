# Higgs Challenge

Higgs Boson particle identification using machine learning project for the EPFL Machine Learning course. The problem was 
orinally proposed by CERN as a Kaggle challenge using a 250 000 samples dataset recorded by the research center. 

In this repo you will find the solution proposed by my team for this challenge


For more information follow this [link](https://www.kaggle.com/c/higgs-boson/overview)

## Overview

Here's a list of the relevant source files 

|Source file|Description|
|---|---|
| `implementations.py`|Regrouping the common machine learning regression algorithms|
|`run.py`|Main script containing the solution of the problem|
|`helpers.py`|Containing all the additional functions used in the project|
|`train_least_squares.py`|Containing the functions used to train the models, specifically for least squares optimization with normal equations|
|`train_least_squares_GD.py`|Containing the functions used to train the models, specifically for least squares optimization using gradient descent or stochastic gradient descent|
|`train_logistic_regression.py`|Containing the functions used to train the models, specifically for logistic regression using gradient descent or stochastic gradient descent|
|`train_reg_logistic_regression.py`|Containing the functions used to train the models, specifically for regularized logistic regression using gradient descent or stochastic gradient descent|
|`train_ridge_regression.py`|Containing the functions used to train the models, specifically for ridge regression using the normal equations|
|`projet1.ipynb`|Notebook of the project with all the visualization and the analysis of the training data and the details about the training of the models|
  

## Requirements

Please make sure you have `Python 3`  with the following packages installed : 

`numpy` 
 
 `matplotlib`  
 



## Instructions

~~~~shell
cd script
python run.py
~~~~
