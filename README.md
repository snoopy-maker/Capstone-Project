# Sparkify User Churn Prediction Model

## Table of Contents
- Installation
- Project Motivation
- File Descriptions
- Instructions
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
1. Load data into Spark and clean data (remove missing values and map data into correct data types) 
2. Exploratory Data Analysis (EDA) using Spark SQL and Spark Dataframe to examine features distribution
3. Feature Engineering - Identify 34 features in the user event dataset that can help to detect pending churn 
4. Build and train models using selected features
5. Predict churn and evaluate performance by comparing 4 MLlib classification models (Logistic regression, Random Forest classifier, Gradient-boosted tree classifier, Linear Support Vector Machine)
  
## Results
The main findings of the code can be found at the post available [here]().
Overall, the f1 score of Gradient-boosted Tree Classifier 0.7941
| Model Name                        | Accuracy | F1-score | Training Time(s) |
| ----------------------------------| -------- |----------| ---------------- |
| Logistic Regression               | 0.7353   | 0.6867   | 427              |
| Gradient-boosted Tree Classifier  | 0.7941   | 0.7564   | 1245             |
| Random Forest Classifier          | 0.7647   | 0.6627   | 379              |
| Linear Support Vector             | 0.7353   | 0.6480   | 5369             |

## Acknowledgements
I would like to express my sincere gratitude to Udacity for the data to complete this educational project.
