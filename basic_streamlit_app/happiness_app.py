import streamlit as st
# Introducing users to the app
st.title("World Happiness App")

st.write("This app explores the World Happiness Report Dataset, looking at factors such as GDP and health in relation to happiness.")
st.write("This includes 158 countries from across the globe in the year 2015.")

# Exploring the dataset
import pandas as pd

st.subheader("Exploring Our Dataset")

# Load the CSV file
df = pd.read_csv("data/2015.csv")

st.write("Here is a preview of our dataset:")
st.dataframe(df)

region = st.selectbox("Select a region", df["Region"].unique())
filtered_df = df[df["Region"] == region]

st.write(f"Countries in region {region}:")
st.dataframe(filtered_df)

# Looking at the relationship between GDP and Happiness Score
st.subheader("GDP vs Happiness Score")

import seaborn as sns
import matplotlib.pyplot as plt
scatter_plot = sns.scatterplot(x = df["Economy (GDP per Capita)"], y = df["Happiness Score"])
st.pyplot(scatter_plot.get_figure())
plt.clf() # Clear the figure to avoid overlap with next plot

# Looking at the relationship between life expectancy and happiness score
st.subheader("Life Expectancy vs Happiness Score")
scatter_plot2 = sns.scatterplot(x = df["Health (Life Expectancy)"], y = df["Happiness Score"])
st.pyplot(scatter_plot2.get_figure())

# Exploring countries by Generosity score and happiness levels
st.subheader("Explore Generosity and Happiness")
generosity_threshold = st.number_input("Select a minimum Generosity score (from 0.0 to 1.0)", min_value = 0.0, max_value = 1.0, value = 0.5, step = 0.01)
filtered_df = df[df["Generosity"] >= generosity_threshold]
st.write(f"Countries with Generosity score above {generosity_threshold}:")
st.dataframe(filtered_df[["Country", "Generosity", "Happiness Score", "Happiness Rank"]])

# Don't forget to run the app with: streamlit run happiness_app.py