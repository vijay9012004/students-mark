# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset from pickle file
pkl_file_path = "students_marks.pkl"  # Make sure you create this file first
with open(pkl_file_path, "rb") as f:
    df = pickle.load(f)

# Prepare data
X = df[['Maths', 'Physics', 'Chemistry']]
y = df['Result']

# Train KNN model
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# Streamlit UI
st.title("Student Marks Grade Predictor (KNN)")
st.write("Enter your marks to predict if your grade is low or high.")

maths = st.number_input("Maths Marks", min_value=0, max_value=100, value=50)
physics = st.number_input("Physics Marks", min_value=0, max_value=100, value=50)
chemistry = st.number_input("Chemistry Marks", min_value=0, max_value=100, value=50)

if st.button("Predict Grade"):
    new_data = np.array([[maths, physics, chemistry]])
    prediction = knn.predict(new_data)
    
    if prediction[0] == 0:
        st.error("Grade is Low ❌")
    else:
        st.success("Grade is High ✅")
