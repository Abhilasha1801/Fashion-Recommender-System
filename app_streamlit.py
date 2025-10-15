import streamlit as st
import numpy as np
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm

# --- Page Config ---
st.set_page_config(page_title="👗 Fashion Recommender", page_icon="🧥", layout="wide")

# --- Custom CSS for better look ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #fdfbfb, #ebedee);
        padding: 2rem;
        border-radius: 1rem;
    }
    h1 {
        color: #E75480;
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
    }
    h2, h3 {
        color: #333333;
    }
    .uploaded-img {
        border: 3px solid #E75480;
        border-radius: 15px;
        padding: 5px;
        background-color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model only once ---
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([base_model, GlobalMaxPooling2D()])
    return model

model = load_model()

# --- Feature extraction function ---
def extract_features(img, model):
    img = img.resize((224, 224)).convert('RGB')
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# --- UI Title ---
st.title("👗 Fashion Recommender System 👗 ")
st.markdown("#### Upload your fashion images to find visually similar styles!")

# --- File Uploader ---
uploaded_files = st.file_uploader("📸 Upload your fashion images", 
                                  type=['jpg', 'jpeg', 'png'], 
                                  accept_multiple_files=True)

if uploaded_files:
    feature_list = []
    images = []
    image_names = []

    progress = st.progress(0)
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            img = Image.open(uploaded_file)
            features = extract_features(img, model)
            feature_list.append(features)
            images.append(img)
            image_names.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        progress.progress((idx + 1) / total)

    st.success(f"✅ Uploaded {len(images)} images successfully!")

    # --- Select query image ---
    st.subheader("🎯 Choose a query image")
    query_name = st.selectbox("Select one image to find similar styles", image_names)
    query_index = image_names.index(query_name)
    query_img = images[query_index]

    st.image(query_img, caption=f"👗 Query Image: {query_name}", use_container_width=False, output_format="JPEG", 
             clamp=True)

    # --- Recommendation logic ---
    def recommend(query_features, feature_list, top_k=5):
        similarities = cosine_similarity([query_features], feature_list)[0]
        indices = np.argsort(similarities)[::-1]
        indices = [i for i in indices if i != query_index][:top_k]
        return indices

    with st.spinner("🔍 Finding visually similar fashion items..."):
        query_features = feature_list[query_index]
        indices = recommend(query_features, feature_list)

    # --- Display similar images ---
    st.subheader("✨ Top 5 Recommended Looks")
    cols = st.columns(5)
    for i, col in zip(indices, cols):
        col.image(images[i], caption=image_names[i], use_container_width=True)

    # --- Save embeddings ---
    st.markdown("#### 💾 Save Extracted Features")
    if st.button("Save Features & Filenames"):
        with open('embeddings.pkl', 'wb') as f:
            pickle.dump(feature_list, f)
        with open('filenames.pkl', 'wb') as f:
            pickle.dump(image_names, f)
        st.success("🎉 Features and filenames saved successfully!")

    st.info(f"📊 Total images processed: {len(images)}")

else:
    st.info("📤 Upload some images to start exploring fashion similarities!")