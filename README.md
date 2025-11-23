# Fashion-Recommender-System
A deep-learning–based fashion recommendation system that suggests visually similar clothing items using image embeddings.
The system uses CNN feature extraction, vector similarity, and a Streamlit web interface for easy interaction.


# Features-

📸 Image-based similarity search

🧠 Feature extraction using Custom CNN

⚡ Fast retrieval using precomputed embeddings

🎨 User interface implemented with Streamlit

🗂️ Dataset of fashion images included

🛠️ Modular and easy-to-extend codebase





# How It Works-

1) The system loads precomputed embeddings (pkl files).

2) User uploads a fashion image.

3) The model extracts the feature vector.

4) Finds the closest embeddings using cosine similarity.

5) Displays the top recommended similar items.



📦 Models Used

CNN-based feature extractor (custom trained or pretrained)

OpenAI CLIP (image encoder)
