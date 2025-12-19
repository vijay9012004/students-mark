import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import kagglehub

# --- Page Config ---
st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓")

# --- Manual KNN Logic ---
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], X_test)
        distances.append((dist, y_train[i]))
    
    # Sort by distance (nearest first)
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Voting logic with tie handling
    counts = {}
    for _, label in k_nearest:
        counts[label] = counts.get(label, 0) + 1
    
    max_count = max(counts.values())
    tied_classes = [key for key, val in counts.items() if val == max_count]
    
    # Randomly select among ties if necessary
    return np.random.choice(tied_classes)

# --- Robust Data Loading ---
@st.cache_data
def load_data():
    pkl_file_path = "students_marks.pkl"
    
    # Try loading from Pickle first
    if os.path.exists(pkl_file_path):
        try:
            with open(pkl_file_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            st.warning("Pickle file incompatible. Downloading fresh data...")
    
    # Fallback: Download from KaggleHub
    path = kagglehub.dataset_download("vicky1999/student-marks-dataset")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    return pd.read_csv(os.path.join(path, csv_file))

df = load_data()
X = df[['Maths', 'Physics', 'Chemistry']].values
y = df['Result'].values

# --- UI Layout ---
st.title("🎓 Student Marks Predictor")
st.write("Enter marks to classify the grade using **Manual K-Nearest Neighbors**.")

# Input Section
col1, col2, col3 = st.columns(3)
with col1:
    maths = st.number_input("Maths", 0, 100, 50)
with col2:
    physics = st.number_input("Physics", 0, 100, 50)
with col3:
    chemistry = st.number_input("Chemistry", 0, 100, 50)

# Slider for dynamic k
k = st.slider("Select number of neighbors (k)", min_value=1, max_value=10, value=3)

# Prediction
if st.button("Predict Grade", use_container_width=True):
    new_point = np.array([maths, physics, chemistry])
    prediction = knn_predict(X, y, new_point, k=k)
    
    st.divider()
    if prediction == 0:
        st.error("### Prediction: LOW GRADE ❌")
    else:
        st.success("### Prediction: HIGH GRADE ✅")

# Data Preview
with st.expander("View Source Data"):
    st.dataframe(df)
