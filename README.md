# Fashion-Recommender-System
A deep-learning–based fashion recommendation system that suggests visually similar clothing items using image embeddings.
The system uses CNN/CLIP feature extraction, vector similarity, and a Streamlit web interface for easy interaction.

🚀 Features

📸 Image-based similarity search

🧠 Feature extraction using:

Custom CNN

CLIP model

⚡ Fast retrieval using precomputed embeddings

🎨 User interface implemented with Streamlit

🗂️ Dataset of fashion images included

🛠️ Modular and easy-to-extend codebase

📁 Project Structure
Fashion Recommender System/
│
├── app.py                      # Main Flask/Streamlit app
├── app_streamlit.py            # Streamlit interface
├── main.py                     # Script to run recommendation logic
│
├── extract_features.py         # Feature extraction using CNN
├── extract_features_clip.py    # Feature extraction using CLIP
│
├── embeddings.pkl              # CNN embeddings
├── embeddings_clip.pkl         # CLIP embeddings
├── features_clip.pkl           # Features used by CLIP
├── filenames.pkl               # Image filenames for CNN embeddings
├── filenames_clip.pkl          # Image filenames for CLIP embeddings
│
├── images/
│   └── images/                 # Dataset of fashion item images
│
└── test.py / try.py            # Testing & experimental scripts

🧰 Installation

Clone the repository:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>


Install dependencies:

pip install -r requirements.txt


If you don’t have a requirements.txt, I can generate one from your project. Just tell me.

▶️ How to Run
Run the Streamlit App
streamlit run app_streamlit.py

Run the Main Application
python app.py

Re-generate Embeddings (Optional)
python extract_features.py
python extract_features_clip.py

🖼️ How It Works

The system loads precomputed embeddings (pkl files).

User uploads a fashion image.

The model extracts the feature vector.

Finds the closest embeddings using cosine similarity.

Displays the top recommended similar items.

📦 Models Used

CNN-based feature extractor (custom trained or pretrained)

OpenAI CLIP (image encoder)
