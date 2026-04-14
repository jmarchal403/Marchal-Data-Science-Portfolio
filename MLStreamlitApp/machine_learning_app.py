import streamlit as st
# Introducing users to the app
st.title("Supervised Machine Learning App")
st.write("This app explores the basics of supervised machine learning.")
st.write("Explore three different datasets (Flight Data, Motor Trends, and Weather Data) using linear regression, decision trees, and k-nearest neighbors.")


# Exploring the datasets
import pandas as pd
st.subheader("Exploring Our Sample Datasets")
# Load the 3 CSV files provided as options for the user
flight_df = pd.read_csv("Data/flights-1m.csv")
motor_df = pd.read_csv("Data/mtcars.csv")
weather_df = pd.read_csv("Data/weather.csv")
# Display a brief preview of the datasets to the user
st.write("Here is a preview of the Flight Data:")
st.dataframe(flight_df.head())
st.write("Here is a preview of the Motor Trends Data:")
st.dataframe(motor_df.head())
st.write("Here is a preview of the Weather Data:")
st.dataframe(weather_df.head())

# Now we will offer the user the option to select one of the datasets for use in linear regression
st.write("Now that you have received a brief introduction to the datasets, please select a dataset from the dropdown menu below to explore further.")
st.write("We will be starting with linear regression, so choose the dataset you would like to use for the linear regression section of the app.")
st.subheader("Linear Regression Analysis")

dataset_choice = st.selectbox("Select a dataset to analyze with linear regression", ["Flight Data", "Motor Trends", "Weather Data"])
if dataset_choice == "Flight Data":
    st.write("You selected the Flight Data. The Flight Data contains information about flights, including flight date, arrival time, departures, delays, and more.")
elif dataset_choice == "Motor Trends":
    st.write("You selected the Motor Trends Data. The Motor Trends Data contains information about various car models, mpg, weight, horsepower, and other performance metrics.")
elif dataset_choice == "Weather Data":
    st.write("You selected the Weather Data. The Weather Data contains information about weather conditions, including temperature, sunshine, rainfall, wind speed, and humidity.")
# Make sure to assign the selected dataframe to a variable for use in the linear regression code section.
if dataset_choice == "Flight Data":
    df = flight_df
elif dataset_choice == "Motor Trends":
    df = motor_df
elif dataset_choice == "Weather Data":
    df = weather_df

# Allow the user to select a target variable and feature variables of their own for the linear regression analysis
column_options = df.columns.tolist()
target_variables = st.selectbox("Select a target variable for analysis", column_options)
feature_variables = st.multiselect("Select feature variables for analysis", column_options)
if set(target_variables) & set(feature_variables):
    st.error("You cannot select the same column as both a target variable and a feature variable. Please re-select.")

# Linear regression code section
# Make sure to import the performance evaluation metrics, the train test split function, and the linear regression model from sklearn - AT THE TOP OF YOUR CODE!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
if st.button("Run Your Linear Regression Analysis"):
    if not target_variables or not feature_variables:
        st.error("Please select at least one target variable and one feature variable to run the analysis.")
    else: 
        X = df[feature_variables]
        y = df[target_variables]
        # We want to drop any missing values from the target/feature variables before training the model
        data = pd.concat([X, y], axis=1).dropna()
        X = data[feature_variables]
        y = data[target_variables]
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.write(f"Linear regression model coefficients: {model.coef_}")
        st.write(f"Linear regression model intercept: {model.intercept_}")
        st.write("You can use the model coefficients and intercept listed here to make predictions based on the feature variables.")
        # This will be what our model actually predicts for the target variable based on the feature variables in the test set
        y_pred = model.predict(X_test)
        # Evaluation code for linear regression model performance
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"R squared: {r2}")
        st.write(f"Root Mean Squared Error: {rmse}")








# Now we will look at our datasets with a decision tree model.
st.subheader("Decision Tree Analysis")
st.write("For the decision tree section of this app, we will be working with the same datasets as before, but we will need to transform the numerical target variables into categorical variables for the deicion tree model we'll build")
st.write("For the Flight Data, we will create a new target variable called 'Delayed' which will indicate whether a flight was delayed (ARR_DELAY > 0) or not (ARR_DELAY <= 0). ")
st.write("For the Weather Data, we will create a new target variable called 'Rain' which will indicate whether there was rainfall (Rainfall > 0) or not (Rainfall <= 0).)")
st.write("For the Motor Trends Data, we will create a new target variable called 'Fuel_Efficient' which will indicate whether a car is fuel efficient (mpg > 30) or not (mpg <= 30).")

# Because the sample datasets work mostly with numerical variables, we need to transform our numerical target variable into a categorical variable for the decision tree model
flight_df["Delayed"] = (flight_df["ARR_DELAY"] > 0).astype(int)
weather_df["Rain"] = (weather_df["Rainfall"] > 0).astype(int)
motor_df["Fuel_Efficient"] = (motor_df["mpg"] > 30).astype(int)

# Offer the user the option to select a different dataset for the decision tree section of the app.
dataset_choice = st.selectbox("Select a dataset to analyze with a decision tree", ["Flight Data", "Motor Trends", "Weather Data"])
if dataset_choice == "Flight Data":
    st.write("You selected the Flight Data. The Flight Data contains information about flights, including flight date, arrival time, departures, delays, and more.")
elif dataset_choice == "Motor Trends":
    st.write("You selected the Motor Trends Data. The Motor Trends Data contains information about various car models, mpg, weight, horsepower, and other performance metrics.")
elif dataset_choice == "Weather Data":
    st.write("You selected the Weather Data. The Weather Data contains information about weather conditions, including temperature, sunshine, rainfall, wind speed, and humidity.")

# Allow the user to select a target variable and feature variables of their own for the decision tree analysis
column_options = df.columns.tolist()
feature_variables = st.multiselect("Select feature variables for analysis", column_options)

# Set the categorical target variable based on the dataset selected by the user
if dataset_choice == "Flight Data":
    target_variable = "Delayed"
elif dataset_choice == "Weather Data":
    target_variable = "Rain"
elif dataset_choice == "Motor Trends":
    target_variable = "Fuel_Efficient"

# Import the decision tree model from sklearn at the top of your code. Make sure to also import the confusion matrix and any other performance evaluation metrics you want to use.
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=4, random_state=42)

# Allow the user to select the max depth of the decision tree with a slider in streamlit. The max depth will impact the performance of the model.
max_depth = st.slider("Select the max depth for the decision tree", min_value=1, max_value=10, value=4)
model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
# Allow the user to select the minimum samples split for the decision tree with a slider in streamlit.
min_samples_split = st.slider("Select the minimum samples split for the decision tree", min_value=2, max_value=20, value=2)
model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
st.write("The max depth paramater will control how many levels the decision tree will have, and the minimum samples split paramater will control the minimum number of samples required to split an internal node.")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

# Decision tree code section
if st.button("Run Your Decision Tree Analysis"):
    if not feature_variables:
        st.error("Please select at least one feature variable to run the analysis.")
    elif target_variable in feature_variables:
        st.error("Your feature variable cannot include the target variable. Please re-select your feature variable.")
    else:
        X = df[feature_variables]
        y = df[target_variable]
        # We want to drop any missing values from the target/feature variables before training the model.
        data = pd.concat([X, y], axis=1).dropna()
        X = data[feature_variables]
        y = data[target_variable]
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        # Again, this will be what our model actually predicts for the target variable based on the feature variables in the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluation code for the performance of the decision tree model
        st.subheader("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.subheader("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the figure to avoid overlap with next plot

        # ROC curve and AUC score for the decision tree model
        st.subheader("ROC Curve:")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()  # Clear the figure to avoid overlap with next plot
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Display the performance metrics in the app.
        st.subheader("Decision Tree Model Performance:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"ROC AUC Score: {roc_auc:.2f}")


# Import the KNN model form sklearn at the top of your code
from sklearn.neighbors import KNeighborsClassifier

st.subheader("K-Nearest Neighbors Analysis")
st.write("Just like the decision tree section, we will be transforming the numerical target variables into categorical variables for the KNN model. We will be using the same target variables we created for the decision tree section of the app (Delayed, Rain, and Fuel_Efficient).")
st.write("However, you still have the option to select the dataset you would like to work with for this section.")

# Allow the user to select a different dataset for the KNN section of the app.
dataset_choice = st.selectbox("Select a dataset to analyze with KNN", ["Flight Data", "Motor Trends", "Weather Data"])
if dataset_choice == "Flight Data":
    st.write("You selected the Flight Data. The Flight Data contains information about flights, including flight date, arrival time, departures, delays, and more.")
elif dataset_choice == "Motor Trends":
    st.write("You selected the Motor Trends data. The Motor Trends data contains information about cars, including their specifications, performance metrics, and more.")
elif dataset_choice == "Weather Data":
    st.write("You selected the Weather Data. The Weather Data contains information about weather conditions, including temperature, humidity, wind speed, and more.")

# Allow the user to select their own feature variables for the KNN analysis
feature_variables = st.multiselect("Select feature variables for KNN analysis", column_options)

# Allow the user select the number of neighbors for the KNN model with a slider in streamlit. The number of neighbors will impact the performance of the model.
k_value = st.slider("Select the number of neighbors (k) for KNN", min_value=1, max_value=20, value=5)
st.write("This will impact the performance of your KNN model, so please experiment with several different values. Be careful of setting a value of k that is too high or too low. Small values can overfit the data, but large values can underfit the data.")
# Allow the user select how to weight the neighbors in the KNN model with a selectbox in streamlit.
weight_option = st.selectbox("Select the weighting method for KNN", ["Uniform", "Distance"])
if weight_option == "Uniform":
    weights = "uniform"
elif weight_option == "Distance":
    weights = "distance"

st.write("This will also impact the performance of your KNN model. The 'uniform' option means that all neighbors will be weighted equally, while the 'distance' option means that closer neighbors will be weighted more heavily than neighbors that are farther away.")

# KNN code section
if st.button("Run Your KNN Analysis"):
    if not feature_variables:
        st.error("Please select at least one feature variable to run the analysis.")
    elif target_variable in feature_variables:
        st.error("Your feature variable cannot include the target variable. Please re-select your feature variable.")
    else:
        X = df[feature_variables]
        y = df[target_variable]
        # We want to drop any missing values from the target/feature variables before training the model.
        data = pd.concat([X, y], axis=1).dropna()
        X = data[feature_variables]
        y = data[target_variable]
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        knn_model = KNeighborsClassifier(n_neighbors=k_value, weights=weights)
        knn_model.fit(X_train, y_train)
        # Again, this will be what our model actually predicts for the target variable based on the feature variables in the test set
        y_pred = knn_model.predict(X_test)
        y_prob = knn_model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        # Display the performance metrics in the app.
        st.subheader("KNN Model Performance:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"ROC AUC Score: {roc_auc:.2f}")
        # Display the classification report for the KNN model
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm)
        st.pyplot(fig_cm)
        plt.clf()  # Clear the figure to avoid overlap with next plot
        # Display the ROC curve for the KNN model
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label='ROC Curve')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curve')
        ax_roc.legend()
        st.pyplot(fig_roc)
        plt.clf()  # Clear the figure to avoid overlap with next plot


st.subheader("Thanks for exploring supervised machine learning with us!")
st.write("We hope that this app has given you the chance to look at a variety of different models and evaluation metrics.")

# Always remember to run the app in streamlit with: streamlit run machine_learning_app.py
# AND make you are working the correct directory in your terminal (MLStreamlitApp) when you run it in streamlit!




