# Machine Learning Streamlit App
This app is designed to walk users through the basics of supervised machine learning. 

Link to the uploaded cloud app: https://8kcm7yeygtscflv8y9ksry.streamlit.app/

Using this app, users can experiment with a variety of different performance evaluation metrics (such as accuracy, precision, confusion matrix, and ROC curve). They can also manipulate the hyperparameters of several different machine learning models, including linear regression, decision trees, and K nearest neighbors.

Three datasets are provided that the user can choose between:
1. Flight Data - data on flights, arrivals, departures, delays, travel time, and more
2. Weather Data - data on rainfall, sunshine, temperature, wind speed, and other weather related factors
3. Motor Trends Data - data on vehicle model and a list of vehicle performance indicators

All of this data can be found at Agents for Data using this link: https://www.agentsfordata.com/csv/sample

### App Layout

I first introduce users to the app, explaining supervised machine learning and outlining the specific functions of the app. I then present users with the option to choose between three sample datasets or upload a dataset of their own.

After the dataset is selected and cleaned, the user can toggle between the three machine learning techniques of this app: linear regression, decision tree analysis, and k nearest neighbors analysis. In each section, users can switch between different target and feature variables, experiment with different hyperparameters, and examine the performance of their model using different metrics and visuals. 

### Tips for Running this App

Make sure to go and look at the code comments I left in the python file if you are confused about any of the commands I used or what the code itself is actually doing!

Otherwise:
1. When uploading the csv files, make sure that the name of the data you downloaded exactly matches the data you are uploading in python.
2. Ensure that you have the correct libraries and models imported. 
    All imports used in this app:  
    import streamlit as st  
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LinearRegression  
    from sklearn.metrics import r2_score, mean_squared_error  
    import numpy as np  
    from sklearn.model_selection import train_test_split  
    from sklearn.tree import DecisionTreeClassifier  
    from sklearn.metrics import accuracy_score, precision_score, roc_auc_score  
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve  
    import matplotlib.pyplot as plt  
    from sklearn.neighbors import KNeighborsClassifier  
4. Do NOT forget to set your working directory to the folder you saved the python file to!

### App Feature Examples 

###### Linear Regression Analysis:
<img width="951" height="806" alt="image" src="https://github.com/user-attachments/assets/c93a648d-9f7f-4abf-8bd5-0ec9734ff381" />

###### Decision Tree Analysis:
<img width="956" height="277" alt="image" src="https://github.com/user-attachments/assets/bf6b3859-5ddb-4592-970d-a8ae21a1f7e9" />

<img width="1021" height="878" alt="image" src="https://github.com/user-attachments/assets/09d722b3-1179-4266-8755-3a20f2e58a91" />


###### KNN Analysis:

<img width="917" height="818" alt="image" src="https://github.com/user-attachments/assets/8b5cd395-95e6-48f3-92dc-d27d56d0ca81" />

<img width="936" height="775" alt="image" src="https://github.com/user-attachments/assets/de96d7e6-5282-4168-9007-c7f35e8353f4" />



#### Want to learn more?
Here is a link to an overview on supervised machine learning. It goes over the basics in more detail, and outlines some of the most popular models: https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/
