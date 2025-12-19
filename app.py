import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Config ---
st.set_page_config(page_title="Student Grade Predictor", page_icon="📝")

# --- Manual KNN Logic ---
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], X_test)
        distances.append((dist, y_train[i]))
    
    # Sort by distance and pick top K
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Voting
    counts = {}
    for _, label in k_nearest:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)

# --- Data Loading ---
pkl_file_path = "students_marks.pkl"

if os.path.exists(pkl_file_path):
    with open(pkl_file_path, "rb") as f: 
        df = pickle.load(f)
    
    X = df[['Maths', 'Physics', 'Chemistry']].values
    y = df['Result'].values

    # --- UI Layout ---
    st.title("🎓 Student Grade Predictor")
    st.write("This app uses a **Manual K-Nearest Neighbors** algorithm to classify student performance.")

    

    st.divider()

    # Input Fields in Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        maths = st.number_input("Maths", 0, 100, 75)
    with col2:
        physics = st.number_input("Physics", 0, 100, 70)
    with col3:
        chemistry = st.number_input("Chemistry", 0, 100, 65)

    if st.button("Predict Grade", use_container_width=True):
        new_data = np.array([maths, physics, chemistry])
        prediction = knn_predict(X, y, new_data, k=3)
        
        st.markdown("### Result:")
        if prediction == 0:
            st.error("Grade Category: **LOW** ❌")
            st.info("Performance is below the dataset average.")
        else:
            st.success("Grade Category: **HIGH** ✅")
            st.info("Excellent! Performance is in the top tier.")

    # Optional: Data Preview
    with st.expander("View Reference Data"):
        st.dataframe(df)

else:
    st.error(f"Error: `{pkl_file_path}` not found. Please ensure your pickle file is in the same folder as app.py.")
