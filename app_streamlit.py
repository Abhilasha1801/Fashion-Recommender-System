import streamlit as st
import pickle
import numpy as np
from PIL import Image
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
import os

# Load saved data
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Feature extraction function
def extract_features(img, model):
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation function
def recommend(features, feature_list):
    distances = np.linalg.norm(feature_list - features, axis=1)
    indices = np.argsort(distances)[:5]
    return indices

# Streamlit UI
st.title("👗 Fashion Recommender System")

uploaded_file = st.file_uploader("Upload an image to find similar items:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Searching for similar images..."):
        query_features = extract_features(img, model)
        indices = recommend(query_features, np.array(feature_list))

    st.subheader("Top 5 Similar Images:")
    cols = st.columns(5)
    for i, col in zip(indices, cols):
        result_img = Image.open(filenames[i])
        col.image(result_img, use_container_width=True)

