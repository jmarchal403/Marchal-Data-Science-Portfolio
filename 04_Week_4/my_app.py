import streamlit as st

# Markdown Hashtag
st.title("Hello, streamlit!")
st.markdown("# Hello, streamlit!")

st.write("This is my first streamlit app.")

if st.button("Click me!"):
    st.write("You clicked the button!")
else:
    st.write("You did not click the button")

### Loading our csv file
import pandas as pd

st.subheader("Exploring Our Dataset")

# Load the CSV file
df = pd.read_csv("data/sample_data-1.csv")

st.write("Here is a preview of our dataset:")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique())
filtered_df = df[df["City"] == city]

st.write(f"People in {city}:")
st.dataframe(filtered_df)

## Add bar chart

st.bar_chart(df["Salary"])

import seaborn as sns

box_plot1 = sns.boxplot(x = df["City"], y = df["Salary"])

st.pyplot(box_plot1.get_figure())