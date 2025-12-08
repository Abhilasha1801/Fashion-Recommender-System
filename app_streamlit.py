import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Try importing FAISS
try:
    import faiss
    use_faiss = True
except ImportError:
    st.warning("⚠️ FAISS not found — using NumPy similarity (slower).")
    use_faiss = False

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="👗 Fashion Recommender AI", layout="wide")

IMAGE_FOLDER = "images/images"  # adjust if needed
FEATURE_FILE = "features_clip.pkl"
MODEL_NAME = "clip-ViT-B-32"

# ---------------------------
# LOAD MODEL & FEATURES
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_features():
    with open(FEATURE_FILE, "rb") as f:
        features = pickle.load(f)
    if isinstance(features, list):
        features = np.array(features)
    return features

model = load_model()
features = load_features()

if use_faiss:
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
else:
    index = None

# ---------------------------
# LOAD IMAGE FILENAMES
# ---------------------------
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(image_files) == 0:
    st.error("🚫 No images found. Please check your folder path.")
    st.stop()

# ---------------------------
# SIMILARITY SEARCH
# ---------------------------
def search_similar(query_vector, top_k=6):
    if use_faiss:
        D, I = index.search(query_vector, top_k)
        return I[0]
    else:
        sims = cosine_similarity(features, query_vector)
        return np.argsort(-sims.flatten())[:top_k]

# ---------------------------
# STYLING
# ---------------------------
st.markdown("""
    <style>


        [data-testid="stAppViewContainer"] {
            background-image: url("https://img.freepik.com/free-photo/abstract-blur-shopping-mall_1203-8772.jpg?semt=ais_se_enriched&w=740&q=80");
            background-size: cover;
            background-position: center;
            color: black;
            background-attachment: fixed;
        }
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url("https://images.unsplash.com/photo-1512436991641-6745cdb1723f?auto=format&fit=crop&w=1920&q=80");
            background-size: cover;
            background-position: center;
            filter: blur(5px);
            z-index: -1;
        }
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
        }
        .stMarkdown {
            background-color: transparent;
            color: black;
        }
        .stTextInput > label > div > p {
            background-color: transparent !important;
            color: black !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
        }
        .stTextInput > label {
            background-color: transparent !important;
            color: black !important;
            font-size: 3rem !important;
            font-weight: 700 !important;
        }
        .stTextInput input {
            font-size: 1rem !important;
            padding: 15px !important;
        }
        .stTextInput, .stButton, .stFileUploader {
            background-color: transparent;
            color: black;
        }

        .title {
            font-size: 2.6rem;
            font-weight: 700;
            text-align: center;
            color: darkblue;
            font-family: 'Poppins', sans-serif;
            margin-bottom: 0.5rem;
        }
        .subtext {
            text-align: center;
            color: black;
            margin-bottom: 2rem;
        }
        .image-card {
            transition: 0.3s;
        }
        .image-card:hover {
            transform: scale(1.03);
            box-shadow: 0 0 10px rgba(255, 64, 129, 0.3);
        }
        .tag {
            display: inline-block;
            background-color: transparent;
            border-radius: 12px;
            padding: 2px 10px;
            margin: 2px;
            font-size: 0.8rem;
            color: white ;
        }
        .chip {
            display: inline-block;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 20px;
            background-color: white;
            cursor: pointer;
            transition: 0.2s;
            font-weight: 500;
        }
        .chip:hover {
            background-color: #FF4081;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>👗 Fashion Recommender AI 👗 </div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Find styles that match your mood, color, and category</div>", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("🎛️ Filters & Options")

mode = st.sidebar.radio("Search Type", ["📝 Text Search", "🖼️ Image Search"])
top_k = st.sidebar.slider("Results per page", 6, 24, 9)

# Color filter chips
st.sidebar.subheader("🎨 Color Filter")
color_options = ["All", "Red", "Black", "White", "Blue", "Green", "Yellow", "Pink"]
selected_color = st.sidebar.radio("", color_options)

# Category dropdown
st.sidebar.subheader("🛍️ Category")
categories = ["All", "Dress", "Top", "Shoes", "Pants", "Jacket", "Skirt"]
selected_category = st.sidebar.selectbox("", categories)

# Pagination setup
page = st.sidebar.number_input("📦 Page", min_value=1, value=1)

# Example queries
#st.sidebar.markdown("##### 💡 Example Queries")
#for example in ["Red dress", "Casual shoes", "Formal jacket", "Summer outfit"]:
#    if st.sidebar.button(example):
#        st.session_state["query"] = example

# ---------------------------
# MAIN AREA
# ---------------------------
if mode == "📝 Text Search":
    query = st.text_input("Search for an outfit:", st.session_state.get("query", ""))
    if st.button("Search") and query.strip() != "":
        with st.spinner("✨ Finding your styles..."):
            query_vector = model.encode([query])
            indices = search_similar(query_vector, top_k * page)
        start_idx = (page - 1) * top_k
        results = indices[start_idx:start_idx + top_k]
        st.subheader(f"Results for **{query}**")

        cols = st.columns(3)
        for i, idx in enumerate(results):
            img_path = os.path.join(IMAGE_FOLDER, image_files[idx])
            filename = os.path.basename(img_path).lower()

            # Simple simulated tags based on filename
            tags = []
            if "dress" in filename: tags.append("Dress")
            if "shoe" in filename: tags.append("Shoes")
            if "top" in filename: tags.append("Top")
            if "pant" in filename or "jean" in filename: tags.append("Pants")
            if "jacket" in filename: tags.append("Jacket")

            # Filter results by color/category
            if selected_color != "All" and selected_color.lower() not in filename:
                continue
            if selected_category != "All" and selected_category.lower() not in filename:
                continue

            with cols[i % 3]:
                st.image(img_path, use_container_width=True, caption=os.path.basename(img_path), output_format="auto")
                tag_html = " ".join([f"<span class='tag'>{t}</span>" for t in tags])
                st.markdown(tag_html, unsafe_allow_html=True)

elif mode == "🖼️ Image Search":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        query_img = Image.open(uploaded).convert("RGB")
        st.image(query_img, caption="Uploaded Image", width=250)
        with st.spinner("🔎 Finding visually similar outfits..."):
            query_vector = model.encode([query_img])
            indices = search_similar(query_vector, top_k * page)
        start_idx = (page - 1) * top_k
        results = indices[start_idx:start_idx + top_k]
        st.subheader("Visually similar recommendations:")

        cols = st.columns(3)
        for i, idx in enumerate(results):
            img_path = os.path.join(IMAGE_FOLDER, image_files[idx])
            filename = os.path.basename(img_path).lower()

            tags = []
            if "dress" in filename: tags.append("Dress")
            if "shoe" in filename: tags.append("Shoes")
            if "top" in filename: tags.append("Top")
            if "pant" in filename or "jean" in filename: tags.append("Pants")
            if "jacket" in filename: tags.append("Jacket")

            # Filter results by color/category
            if selected_color != "All" and selected_color.lower() not in filename:
                continue
            if selected_category != "All" and selected_category.lower() not in filename:
                continue

            with cols[i % 3]:
                st.image(img_path, use_container_width=True, caption=os.path.basename(img_path), output_format="auto")
                tag_html = " ".join([f"<span class='tag'>{t}</span>" for t in tags])
                st.markdown(tag_html, unsafe_allow_html=True)

st.markdown("---")
st.caption("💖 Powered by CLIP + Streamlit + FAISS | Designed by AI")
