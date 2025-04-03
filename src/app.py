import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os

# Title page
st.title('Movie Recommendation System')

# Load the trained KNN model
model_path = "../models/knn_neighbors-7_algorithm-auto_metric-cosine_leaf_size-40_radius-1.0.sav"
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the dataset
movies_df = pd.read_csv("../data/processed/processed_data.csv") 
movie_titles = movies_df["title"].tolist()

# Vectorize movie tags
vector = TfidfVectorizer()
matrix = vector.fit_transform(movies_df["tags"])

# Normalize titles for matching
movie_titles_lower = [title.strip().lower() for title in movie_titles]

# Movie selection dropdown
movie_title = st.selectbox(label='Select a movie title:', options=movies_df['title'], placeholder='Avatar')

# Normalize input
movie_title = movie_title.strip().strip('"').strip("'").lower()

def index():
    recommendations = None
    error = None

    if movie_title in movie_titles_lower:
        # Get original index (before lowercasing)
        matched_index = movies_df[movies_df["title"].str.strip().str.lower() == movie_title].index
                
        if matched_index.empty:
            error = "Movie not found. Please try another title."
        else:
            movie_index = matched_index.tolist()[0]  # Extract index

            # Find the nearest neighbors
            distances, indices = model.kneighbors(matrix[movie_index])

            # Get recommended movie titles (excluding the first one, which is the input movie)
            recommendations = [movies_df.iloc[i]["title"] for i in indices[0][1:]]

            # Display recommendations
            st.subheader("Recommended Movies:")
            st.write("\n".join(recommendations))
    else:
        error = "Movie not found. Please try another title."

    if error:
        st.error(error)

# Call the function
index()

# Streamlit link: https://dansah2-streamlitwebapp-machine-learning.onrender.com