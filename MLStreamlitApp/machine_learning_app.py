
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (r2_score, mean_squared_error,accuracy_score, precision_score,confusion_matrix, ConfusionMatrixDisplay,roc_curve, roc_auc_score)
def colored_box(text, color):
    st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px;">{text}</div>', unsafe_allow_html=True)
# Introducing users to the app
st.title("Supervised Machine Learning App")
st.write("Welcome! This app explores the basics of supervised machine learning.")
st.divider()



st.header("What is Supervised Machine Learning?")
st.write("Supervised machine learning is a type of machine learning where the model is trained on a labeled dataset, meaning that the input data is paired with the correct output labels. The goal is for the model to learn the association between certain features and correct labels in order to make accurate predictions on new, unseen data.")

st.subheader("What does this app do specifically?")
st.write("This app will allow you to explore a dataset of your own or choose a sample dataset provided to explore three common supervised machine learning models:")
st.write("1. Linear Regression")
st.write("2. Decision Tree")
st.write("3. K-Nearest Neighbors (KNN)")
st.write("For each supervised machine learning method, you will have the ability to select the target variable and feature variables of your choice, as well as experiment with different hyperparameters. You will also be able to evaluate the performance of your model using different metrics and visualizations.")
st.divider()



# Exploring the datasets
import pandas as pd
st.subheader("Choosing Your Dataset")

# Allow the user to either upload their own dataset or select one of the sample datasets provided in the app. The sample datasets are flights-1m.csv, mtcars.csv, and weather.csv.
dataset_option = st.selectbox("Select a dataset to explore", ["Upload Your Own Dataset", "Flight Data", "Motor Trends", "Weather Data"])
df = None  

if dataset_option == "Upload Your Own Dataset":
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")
    if uploaded_file is None:
        st.warning("Please upload a CSV file to explore your own dataset.")
        st.stop()
    else:
        df = pd.read_csv(uploaded_file)
        st.write("Here is a preview of your uploaded dataset:")
        st.dataframe(df.head())

elif dataset_option == "Flight Data":
    st.write("You selected the Flight Data. The Flight Data contains information about flights, including flight date, arrival time, departures, delays, and more.")
    df = pd.read_csv("MLStreamlitApp/Data/flights-1m.csv")
    st.dataframe(df.head())

elif dataset_option == "Motor Trends":
    st.write("You selected the Motor Trends Data. The Motor Trends Data contains information about various car models, mpg, weight, horsepower, and other performance metrics.")
    df = pd.read_csv("MLStreamlitApp/Data/mtcars.csv")
    st.dataframe(df.head())

elif dataset_option == "Weather Data":
    st.write("You selected the Weather Data. The Weather Data contains information about weather conditions, including temperature, sunshine, rainfall, wind speed, and humidity.")
    df = pd.read_csv("MLStreamlitApp/Data/weather.csv")
    st.dataframe(df.head())

categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first = True)
df_encoded = df_encoded.dropna()

# To ensure that no file is too large and overwhelms the app
if len(df_encoded) > 500:
    df_encoded = df_encoded.head(500)

st.subheader("Don't forget to clean!")
st.write("The dataset has been cleaned by dropping rows with missing values and converting categorical variables into dummy variables. Here's a preview of the cleaned dataset:")
st.dataframe(df_encoded.head())
colored_box("Feedback: Cleaning is an important step in supervised machine learning because many models cannot handle missing values or raw text", color ="#34d16b")
    
st.divider()





# END OF INTRODUCTION

# END OF INTRODUCTION

# END OF INTRODUCTION



st.header("Exploring Supervised Machine Learning Techniques")
# Now that they have been given a brief introduction to the datasets, we will give them the option to pick a supervised machine learning technique to explore
technique_options = st.selectbox("Select a supervised machine learning technique to explore", ["Linear Regression", "Decision Tree", "K-Nearest Neighbors (KNN)"])  


# Linear Regression code section
# Make sure to import the performance evaluation metrics, the train test split function, and the linear regression model from sklearn - AT THE TOP OF YOUR CODE!
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split


if technique_options == "Linear Regression":
    st.write("Great! You selected linear regression. Linear regression is a technique in supervised machine learning that examines the relationship between one target variable and one or more feature variables.")
    with st.expander("How it works (in more detail):"):
        colored_box("Linear regression works by finding the best-fitting line that minimizes the sum of squared differences between observed values and predicted values. When you use linear regression, the model will spit out coefficients for each feature variable and an intercept. These coefficients tell you the impact of each feature variable on the target variable (while holding all other feature variables constant). The intercept is the expected value of your target variable when all feature variables are at zero. You can use the coefficients and intercept make predictions based on the feature variables", color = "#dac17f")

    # Allow the user to select a target variable and feature variables of their own for the linear regression analysis
    column_options = df_encoded.columns.tolist()
    target_variables = st.selectbox("Select a target variable for analysis", column_options)
    feature_variables = st.multiselect("Select feature variables for analysis", column_options)
    if target_variables in feature_variables:
        st.error("You cannot select the same column as both a target variable and a feature variable. Please re-select.")
    elif not target_variables or not feature_variables:
        st.warning("Please select at least one target variable and one feature variable to run the analysis.")
    else:
        X = df_encoded[feature_variables]
        y = df_encoded[target_variables]
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
        # This will be what our model actually predicts for the target variable based on the feature variables in the test set
        y_pred = model.predict(X_test)
        # Evaluation code for linear regression model performance
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"R squared: {r2}")
        st.write(f"Root Mean Squared Error: {rmse}")
        colored_box("Feedback: R squared is a measure of how well the model explains the variance in the target variable. A higher R squared value (closer to 1 than 0) indicates a better fit of the model to the data. RMSE is a measure of the average distance between the predicted values and the actual values. A lower RMSE value indicates better performance of the model. The coefficients tell us the impact of one additional unit increase from each feature variable on our target variable. The intercept is the value of our target variavble when all selected feature variables are at zero", color="#34d16b")
    st.divider()
    st.subheader("Conclusion")
    st.write("That concludes the linear regression section of this app. However, I encourage you to keep experimenting with different feature and target variables to discover any surprising relationships and reduce the RMSE. You are also welcome to toggle to the other supervised machine learning methods in this app.")
    st.write("If you would like to learn more about linear regression in machine learning, please feel free to visit this page from GeeksForGeeks: https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/")



# END OF LINEAR REGRESSION SECTION

# END OF LINEAR REGRESSION SECTION

# END OF LINEAR REGRESSION SECTION




# Decision Tree code section
elif technique_options == "Decision Tree":
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score, precision_score
    st.write("Great! You selected decision tree. A decision tree is a type of supervised machine learning model that uses a tree-like structure to make decisions based on the feature variables. The model splits the data into branches based on the values of the feature variables, and each branch represents a possible outcome or decision.")
    with st.expander("How it works (in more detail):"):
            colored_box("The decision tree algorithm works by splitting the data into subsets based on the feature variables. The algorithm selects the feature variable and the corresponding threshold that best separates the data into different classes (for classification) or minimizes the variance (for regression). This process continues until a stopping criterion is met, such as reaching a maximum depth or having a minimum number of samples in a leaf node. The resulting tree structure can be used to make predictions by traversing the tree from the root to a leaf node based on the values of the feature variables.", color="#dac17f")    
    # Now, we should allow the user to select a target variable and feature variables of their own for the decision tree analysis. However, because the sample datasets work mostly with numerical variables, we need to transform our numerical target variable into a categorical variable for the decision tree model. We will create new categorical target variables for each dataset to use in the decision tree section of the app.
    st.write("For the decision tree section of this app, we will be working with the same datasets as before, but we will need to transform the numerical target variables into categorical variables for the deicion tree model we'll build:")
    colored_box("For the Flight Data, we will create a new target variable called 'Delayed' which will indicate whether a flight was delayed (ARR_DELAY > 0) or not (ARR_DELAY <= 0). ", color="#a3c1e3")
    colored_box("For the Weather Data, we will create a new target variable called 'Rain' which will indicate whether there was rainfall (Rainfall > 0) or not (Rainfall <= 0).)", color="#a3c1e3")
    colored_box("For the Motor Trends Data, we will create a new target variable called 'Fuel_Efficient' which will indicate whether a car is fuel efficient (mpg > 30) or not (mpg <= 30).", color="#a3c1e3")
    colored_box("If you have uploaded your own dataset, your options will be limited to the categorical variables available in your dataset.", color="#a3c1e3")
    if dataset_option == "Flight Data" and "ARR_DELAY" in df_encoded.columns:
        df_encoded["Delayed"] = df_encoded["ARR_DELAY"].apply(lambda x: 1 if x > 0 else 0)
    elif dataset_option == "Weather Data" and "Rainfall" in df_encoded.columns:
        df_encoded["Rain"] = df_encoded["Rainfall"].apply(lambda x: 1 if x > 0 else 0)
    elif dataset_option == "Motor Trends" and "mpg" in df_encoded.columns:
        df_encoded["Fuel_Efficient"] = df_encoded["mpg"].apply(lambda x: 1 if x > 30 else 0)
    elif dataset_option == "Upload Your Own Dataset":
        st.write("Since you uploaded your own dataset, we have not created a new target variable for you. Please make sure to select a categorical target variable for the decision tree section of the app.")
    
    # Allow the user to select a target variable and feature variables of their own for the decision tree analysis
    if dataset_option == "Flight Data":
        target_options = ["Delayed"]
    elif dataset_option == "Weather Data":
        target_options = ["Rain"]
    elif dataset_option == "Motor Trends":
        target_options = ["Fuel_Efficient"]
    elif dataset_option == "Upload Your Own Dataset":
        target_options = df.select_dtypes(exclude=['number']).columns.tolist()
        for col in df.columns:
            if df[col].nunique() <= 10 and col not in target_options:
                target_options.append(col)
    else:
        target_options = []
    
    if len(target_options) == 0:
        st.warning("No categorical variables found in the dataset. Please select a different dataset or upload a new dataset with categorical variables to use as the target variable for the decision tree analysis.")
        st.stop()
    
    target_variable = st.selectbox("Select a target variable for analysis", target_options)
    feature_variables = st.multiselect("Select feature variables for analysis", [col for col in df_encoded.columns if col != target_variable])

    
    max_depth = st.slider("Select the max depth for the decision tree", min_value=1, max_value=15, value=4)
    colored_box("Guide: The max depth of the decision tree is a hyperparameter that decides how many splits the tree will make before it stops. A lower max depth will have fewer splits but may underfit the data. A higher max depth will have more splits but may end up overfitting the data. You should experiment with several different values to see how it impacts the model performance.", color="#e7ea9e")
    min_samples_split = st.slider("Select the minimum samples split for the decision tree", min_value=2, max_value=20, value=2)
    colored_box("Guide: The minimum samples split is another decision tree hyperparameter that determines when the model is allowed to split a node. It sets a minimum requirement of data points in a node before it can even be split. Lowering the minimum may lead to overfitting, while setting the minimum too high may underfit the data. Again, we recommend experimenting with this hyperparameter to see how it impacts performance", color = "#e7ea9e")
    st.write()
    if target_variable in feature_variables:
        st.error("You cannot select the same column as both a target variable and a feature variable. Please re-select.")
    elif not target_variable or not feature_variables:
        st.warning("Please select at least one target variable and one feature variable to run the analysis.")
    else:
        if st.button("Run Your Decision Tree Analysis"):
            X = df_encoded[feature_variables]
            y = df_encoded[target_variable]
            # We want to drop any missing values from the target/feature variables before training the model.
            data = pd.concat([X, y], axis=1).dropna()
            X = data[feature_variables]
            y = data[target_variable]
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.subheader("Decision Tree Results:")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average ='weighted', zero_division=0)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            # Now provide feedback/guide on what accuracy and precision scores mean
            colored_box("Guide: Accuracy scores record the portion of all predictions that our model correctly predicted. Precision scores tell us the portion of positives our model correctly predicted.", color = "#e7ea9e")
            if y.unique().shape[0] == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            st.subheader("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig_cm, ax_cm = plt.subplots()
            disp.plot(ax=ax_cm, cmap=plt.cm.Blues)
            st.pyplot(fig_cm)
            colored_box("Guide: The Confusion Matrix gives us a visualization of where the model is predicting correctly, and where it is predicting incorrectly. In the top left, we can see how many negatives were correctly identified. In the bottom right, we can see how many positives were correctly identified", color = "#e7ea9e")
            plt.clf()  # Clear the figure to avoid overlap with next plot
            if y.nunique() == 2:
                st.subheader("ROC Curve:")
                fpr,tpr, thresholds = roc_curve(y_test, y_prob)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label='ROC Curve')
                ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve')
                ax_roc.legend()
                st.pyplot(fig_roc)
                st.write(f"ROC AUC Score: {roc_auc:.2f}")
                colored_box("Guide: The ROC Curve presents the tradeoff between true positives and true negatives. When the curve is farther away from the straight line and curves further into the top left portion of the graph, it is associated with better model performance. The ROC AUC score condenses this performance into one number. Scores closer to 1 indicate that the model is doing well at correctly classifying datapoints. A score around 0.5 means that the model is doing about as well as it would if it guessed randomly.", color = "#e7ea9e")
        
                plt.clf()  # Clear the figure to avoid overlap with next plot
            st.subheader("Decision Tree Visualization:")
            fig_tree, ax_tree = plt.subplots(figsize=(12, 8))
            plot_tree(model, feature_names=feature_variables, class_names=[str(cls) for cls in model.classes_], filled=True, ax=ax_tree)
            st.pyplot(fig_tree)
            # Now provide feedback on the 
            colored_box("Guide: The decision tree visualization tells us which feature variables were used to split the data. It functions as a visual of the logical decision making process that the model goes through when classifying the data. The splits at the top of the tree are typically the most important while splits further down the tree are usually smaller and more specific.", color = "#e7ea9e")
            plt.clf()  # Clear the figure to avoid overlap with next plot
    st.divider()
    st.subheader("Conclusion")
    st.write("That concludes the Decision Tree section of this app. However, I encourage you to keep experimenting with different variables, max depths, and minimum samples split. You are also welcome to toggle to the other supervised machine learning methods if you haven't done so.")
    st.write("If you would like to learn more about decision trees, feel free to check out this page from the DEV Community site: https://dev.to/arbashhussain/a-step-by-step-guide-to-decision-trees-in-machine-learning-3h8h")


    

    # Import the KNN model form sklearn at the top of your code
    from sklearn.neighbors import KNeighborsClassifier
elif technique_options == "K-Nearest Neighbors (KNN)":
    st.write("Great! You selected K-Nearest Neighbors (KNN). KNN is a machine learning method that classifies each data point based on other nearby data points that are similar.")
    
    with st.expander("How it works (in more detail)"):
        colored_box("KNN works by finding the k closest observations to a new data point. KNN will then classify each subsequent data point based on the class most common among its surrounding neighbors", color = "#dac17f")
        # Now, we should allow the user to select a target variable and feature variables of their own for the decision tree analysis. However, because the sample datasets work mostly with numerical variables, we need to transform our numerical target variable into a categorical variable for the decision tree model. We will create new categorical target variables for each dataset to use in the decision tree section of the app.
    st.write("For the K-Nearest Neighbors section of this app, we will need to transform the numerical target variables into categorical variables:")
    colored_box("For the Flight Data, we will create a new target variable called 'Delayed' which will indicate whether a flight was delayed (ARR_DELAY > 0) or not (ARR_DELAY <= 0). ", color="#679edc")
    colored_box("For the Weather Data, we will create a new target variable called 'Rain' which will indicate whether there was rainfall (Rainfall > 0) or not (Rainfall <= 0).)", color="#679edc")
    colored_box("For the Motor Trends Data, we will create a new target variable called 'Fuel_Efficient' which will indicate whether a car is fuel efficient (mpg > 30) or not (mpg <= 30).", color="#679edc")
    colored_box("If you have uploaded your own dataset, your options will be limited to the categorical variables available in your dataset.", color="#679edc")
    if dataset_option == "Flight Data" and "ARR_DELAY" in df_encoded.columns:
        df_encoded["Delayed"] = df_encoded["ARR_DELAY"].apply(lambda x: 1 if x > 0 else 0)
    elif dataset_option == "Weather Data" and "Rainfall" in df_encoded.columns:
        df_encoded["Rain"] = df_encoded["Rainfall"].apply(lambda x: 1 if x > 0 else 0)
    elif dataset_option == "Motor Trends" and "mpg" in df_encoded.columns:
        df_encoded["Fuel_Efficient"] = df_encoded["mpg"].apply(lambda x: 1 if x > 30 else 0)
    elif dataset_option == "Upload Your Own Dataset":
        st.write("Since you uploaded your own dataset, we have not created a new target variable for you. Please make sure to select a categorical target variable for the decision tree section of the app.")
    
    if dataset_option == "Flight Data":
        target_options = ["Delayed"]
    elif dataset_option == "Weather Data":
        target_options = ["Rain"]
    elif dataset_option == "Motor Trends":
        target_options = ["Fuel_Efficient"]
    elif dataset_option == "Upload Your Own Dataset":
        target_options = df.select_dtypes(exclude = ["number"]).columns.tolist()

        for col in df.columns:
            if df[col].nunique() <=10 and col not in target_options:
                target_options.append(col)
    if len(target_options) ==0:
        st.warning("No categorical variables found. Please select a different dataset or upload a dataset with categorical variables!")
        st.stop()
    target_variable = st.selectbox("Select a target variable for KNN analysis", target_options)

    feature_options = [col for col in df_encoded.select_dtypes(include=["number"]).columns if col != target_variable]
    feature_variables = st.multiselect("Select feature variables for KNN analysis", feature_options)


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
        X = df_encoded[feature_variables]
        y = df_encoded[target_variable]
        # We want to drop any missing values from the target/feature variables before training the model.
        data = pd.concat([X, y], axis=1).dropna()
        X = data[feature_variables]
        y = data[target_variable]
        # Split the data into training and testing sets
        if X.shape[1] ==0:
            st.error("No usable numeric feature variables were selected. Please select at least one numeric feature variable.")
            st.stop()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        knn_model = KNeighborsClassifier(n_neighbors=k_value, weights=weights)
        knn_model.fit(X_train, y_train)
        # Again, this will be what our model actually predicts for the target variable based on the feature variables in the test set
        y_pred = knn_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average = "weighted", zero_division=0)
        if y.nunique() ==2:
            y_prob = knn_model.predict_proba(X_test)[:,1]
            roc_auc = roc_auc_score(y_test,y_prob)
        # Display the performance metrics in the app.
        st.subheader("KNN Model Performance:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        if y.nunique() == 2:
            st.write(f"ROC AUC Score: {roc_auc:.2f}")
            colored_box("Guide: Accuracy scores record the portion of all predictions that our model correctly predicted. Precision scores tell us the portion of positives our model correctly predicted. The ROC AUC score measures how well the model classifies positive instances compared to negative instances. A score at 0.5 is equivalent to random guessing. The closer the score is to 1, the better the performance of the model.", color = "#e7ea9e")
        else:
            colored_box("Guide: Accuracy scores record the portion of all predictions that our model correctly predicted. Precision scores tell us the portion of positives our model correctly predicted.", color = "#e7ea9e")
            st.write("ROC AUC Score is only shown for binary classification problems")
        # Display the classification report for the KNN model
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm)
        st.pyplot(fig_cm)
        colored_box("Guide: The Confusion Matrix gives us a visualization of where the model is predicting correctly, and where it is predicting incorrectly. In the top left, we can see how many negatives were correctly identified. In the bottom right, we can see how many positives were correctly identified", color = "#e7ea9e")

        plt.clf()  # Clear the figure to avoid overlap with next plot
        # Display the ROC curve for the KNN model
        if y.nunique() ==2:
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
            colored_box("Guide: The ROC Curve presents the tradeoff between true positives and true negatives. When the curve is farther away from the straight line and curves further into the top left portion of the graph, it is associated with better model performance.", color = "#e7ea9e")
        
            plt.clf()  # Clear the figure to avoid overlap with next plot
    st.divider()
    st.subheader("Conclusion")
    st.write("That concludes the K-Nearest Neighbors section of this app. However, I encourage you to keep experimenting with different variables, number of neighbors, and weighting methods. You can also toggle to different supervised machine learning methods in this app if you haven't done so.")
    st.write("If you would like to learn more about K-Nearest Neighbors, feel free to explore this page from GeeksForGeeks: https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/")



# Always remember to run the app in streamlit with: streamlit run machine_learning_app.py
# AND make you are working the correct directory in your terminal (MLStreamlitApp) when you run it in streamlit!




