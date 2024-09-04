
Model Pipeline Class

Overview

This repository contains a Python module designed for building and evaluating machine learning models. The module features two main classes: a Regression Pipeline and a K-Nearest Neighbors (KNN) Classification Pipeline. These classes encapsulate the entire process, from data preprocessing to model evaluation, providing a streamlined and reusable approach to model development.

Installation

Ensure that you have the necessary Python packages installed. You can install them using pip:

pip install pandas matplotlib seaborn scikit-learn numpy

Libraries

The following libraries are used in this module:

- pandas: For data manipulation and analysis.
- matplotlib.pyplot: For plotting and visualizing data.
- seaborn: For statistical data visualization.
- sklearn.model_selection: For splitting data into training and test sets, performing cross-validation, and hyperparameter tuning.
- sklearn.neighbors: For implementing the K-Nearest Neighbors classifier.
- sklearn.metrics: For calculating model performance metrics like accuracy, confusion matrix, and classification report.
- sklearn.preprocessing: For standardizing features by removing the mean and scaling to unit variance.
- sklearn.pipeline: For creating a pipeline that automates the workflow of machine learning models.
- numpy: For numerical computations and array manipulation.

Class Overview

1. Regression Pipeline

This class is designed to streamline the process of setting up and evaluating regression models. It includes:

- Data preprocessing steps, such as feature scaling and train-test splitting.
- Implementation of different regression models like Linear Regression, Lasso, Ridge, and ElasticNet.
- Automated hyperparameter tuning using cross-validation.
- Metrics evaluation to assess the performance of the models.

2. KNN Classification Pipeline

This class provides a framework for building, tuning, and evaluating a K-Nearest Neighbors classifier. Key features include:

- Data preprocessing, including standardization and train-test splitting.
- Automatic determination of the optimal number of neighbors using grid search with cross-validation.
- Comprehensive model evaluation with accuracy scores, confusion matrices, and classification reports.

Usage

Regression Pipeline

from your_module_name import RegressionPipeline

# Initialize the pipeline
regression_pipeline = RegressionPipeline()

# Fit the model
regression_pipeline.fit(X_train, y_train)

# Evaluate the model
regression_pipeline.evaluate(X_test, y_test)

KNN Classification Pipeline

from your_module_name import KnnClassificationPipeline

# Initialize the pipeline
knn_pipeline = KnnClassificationPipeline()

# Fit the model
knn_pipeline.fit(X_train, y_train)

# Evaluate the model
knn_pipeline.evaluate(X_test, y_test)
