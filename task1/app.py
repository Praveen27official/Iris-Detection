import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Title
st.title("🌸 Iris Flower Classification Web App")

# Load Dataset
df = pd.read_csv("Iris.csv")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Prepare data
X = df.drop(["Species", "Id"], axis=1)
y = df["Species"]

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# User input
st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.5)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)

# Output
st.subheader("Prediction Result:")
st.success(f"The flower is: {prediction[0]}")