# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load dataset from pickle
with open("students_marks.pkl", "rb") as f:
    df = pickle.load(f)

# Prepare data
X = df[['Maths', 'Physics', 'Chemistry']].values
y = df['Result'].values

# Manual KNN functions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], X_test)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    counts = {}
    for _, label in k_nearest:
        counts[label] = counts.get(label, 0) + 1
    # Return the label with the highest count
    return max(counts, key=counts.get)

# Streamlit UI
st.title("Student Marks Grade Predictor (Manual KNN)")
st.write("Enter your marks to predict if your grade is low or high.")

maths = st.number_input("Maths Marks", min_value=0, max_value=100, value=50)
physics = st.number_input("Physics Marks", min_value=0, max_value=100, value=50)
chemistry = st.number_input("Chemistry Marks", min_value=0, max_value=100, value=50)

if st.button("Predict Grade"):
    new_data = np.array([maths, physics, chemistry])
    prediction = knn_predict(X, y, new_data, k=3)
    
    if prediction == 0:
        st.error("Grade is Low ❌")
    else:
        st.success("Grade is High ✅")
