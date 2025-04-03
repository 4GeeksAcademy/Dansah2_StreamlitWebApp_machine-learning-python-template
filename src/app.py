import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os

# Load the trained KNN model
model_path = "../models/knn_neighbors-7_algorithm-auto_metric-cosine_leaf_size-40_radius-1.0.sav"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the dataset
movies_df = pd.read_csv("../data/processed/processed_data.csv") 
movie_titles = movies_df["title"].tolist()