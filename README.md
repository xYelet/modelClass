# Model Pipeline Class

## Overview

This repository contains a Python module designed for building and evaluating machine learning models. The module features two main classes: a Regression Pipeline and a K-Nearest Neighbors (KNN) Classification Pipeline. These classes encapsulate the entire process, from data preprocessing to model evaluation, providing a streamlined and reusable approach to model development.

## Installation

Ensure that you have the necessary Python packages installed. You can install them using pip:

```pip install pandas matplotlib seaborn scikit-learn numpy```

### Libraries

The following libraries are used in this module:

- pandas: For data manipulation and analysis.
- matplotlib.pyplot: For plotting and visualizing data.
- seaborn: For statistical data visualization.
- sklearn: For using learning models.
- numpy: For numerical computations and array manipulation.

## Class Overview

1. Regression

This class is designed to streamline the process of setting up and evaluating regression models

2. KNN Classification

This class provides a framework for building, tuning, and evaluating a K-Nearest Neighbors classifier. Key features include:

- Doing a normal knn Classification with 3 neighbours normally.
- Finding the best parameter combination for knn-usage with knnBestFit.
- Finding the best neighbout numbers for knn with knnBestNeighbout.
- If you have a list of data frame, you can train the data on the first 2/3 of the lists and apply it on the last 1/3. This ensure that we do not train and test the data of the same df

## Usage
