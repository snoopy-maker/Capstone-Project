# Sparkify User Churn Prediction Model

## Table of Contents
- Installation
- Project Motivation
- File Descriptions
- Instructions
- Project Steps
- Results
- Acknowledgements

## Installation
Software requirements: 
* Anaconda distribution of Python versions 3.*
* Spark v2.4.3

The following packages (libraries) need to be installed to run the code here beyond the Anaconda distribution of Python. 
* pyspark.ml package
* pyspark.mllib package
* pyspark.sql module
* Pandas, NumPy, matplotlib, seaborn

## Project Motivation
The Captone Project is the final project of Udacity Data Scientist Nanodegree. In this Capstone project, users event dataset is manipulated with Spark to build a maching learning model for predicting churn rates in **Sparkify**, a fictional music streaming service platform created by Udacity just as Spotify. Users stream songs in Sparkify using free or premium subscription with monthly flat rate charged. They can upgrade, downgrade or cancel their service at anytime.

## File Descriptions
  - sparkify.ipynb : This notebook has all the functions for processing the data and ml pipeline.
  - mini_sparkify_event_data.json : A small subset of dataset used for this analysis.

You may analyze full dataset (12GB, avaliable at Udacity Amazon s3 bucket: s3n://udacity-dsnd/sparkify/sparkify_event_data.json) using AWS or IBM Cloud.

## Project Steps
1. Load data into Spark and clean data (remove missing values, map data into correct data types by adding new variables) 
2. Exploratory Data Analysis (EDA) using Spark SQL and Spark Dataframe to examine features distribution
3. Feature Engineering - Identify 33 features in the user event dataset that can help to detect pending churn 
4. Build and train models using selected features
5. Predict churn and evaluate performance by comparing 4 MLlib classification models (Logistic regression, Random Forest classifier, Gradient-boosted tree classifier, Linear Support Vector Machine)
  
## Results
At first, we perform prediction by running all models with default parameters and obtain the following result. F1 scores for all models are relatively close, between 0.7262 and 0.7797.

| Model Name                        | Accuracy | F1-score | Training Time(s) |
| ----------------------------------| -------- |----------| ---------------- |
| Logistic Regression               | 0.8235   | 0.7797   | 1037             |
| Gradient-boosted Tree Classifier  | 0.7339   | 0.7647   | 1587             |
| Random Forest Classifier          | 0.6627   | 0.7647   | 588              |
| Linear Support Vector             | 0.7941   | 0.7262   | 3774             |

Obviously, Logistic Regression model achieves the best performance, with an accuracy of 0.8235 and a F1-Score of 0.7797 although it takes slightly longer time  about 1037 seconds to train the model compared to Random Forest Classifier. We did a hyperparameter tuning by performing a grid search with different values for the following parameters with the hope of improving accuracy and F1-score.

* regParam(0.01, 0.1, 1.0)
* elasticNetParam (0.0, 0.5, 1.0)
* maxIter (10, 20, 30)

Unfortunately, the accurancy and F1-score reduced to 0.7647 and 0.7062 after lowering maxIter from 100 to 30  as the model may not have enough iterations to converge to the optimal set of coefficients. In addition, the running cost of this tuned model increased significantly with triple training time compared to initial model.  Hence, we conclude that we should stick to its default parametes while building Logistic Regression model for this small dataset.

The following is the final evaluation result of tuned Logistic Regression model. 

| Model Name              | Accuracy | F1-score | Training Time(s) |
| ------------------------| -------- |----------| ---------------- | 
| Logistic Regression     | 0.7647   | 0.7062   | 3242             |

More details of analysis can be found at the post available [here](https://medium.com/@swetling.chan/sparkify-user-churn-prediction-model-064b3afdc33c).

## Acknowledgements
I would like to express my sincere gratitude to Udacity for the data to complete this educational project.
