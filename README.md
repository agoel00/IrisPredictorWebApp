# Iris Species Prediction Web App
## Flask web app which predicts the species of Iris flower.

You can find a live version of the site [here](http://agoel00.pythonanywhere.com)

<img src="demo.gif" height="450" width="250">

## Introduction 

It uses 3 trained ML models - 
1. Logistic Regression
2. K Nearest Negihbours
3. SVM

These models have been trained on my local machine and saved using pickle library in Python. Then the saved models are called using Flask.

BulmaCSS is used for the frontend.

## Requirements
1. flask
2. scikit-learn
3. pandas
4. numpy

## Instructions

To run the app locally, enter these commands in your terminal: 

> git clone https://github.com/agoel00/IrisPredictorWebApp

This downloads the repository from Github to your local machine

> cd IrisPredictorWebApp

Change your current working directory to this

> pip install requirements.txt

Install the required libraries

> python app.py

Run the app and predict away! :)

