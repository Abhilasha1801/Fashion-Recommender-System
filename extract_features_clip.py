import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ✅ Load CLIP model (ViT-B/32 is a good balance of speed and accuracy)
model = SentenceTransformer('clip-ViT-B-32')

# 📁 Set your image folder
folder_path = "images/images"

# Lists for features and filenames
feature_list = []
filenames = []

# 🔍 Iterate over images and extract CLIP embeddings
for file in tqdm(os.listdir(folder_path)):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(folder_path, file)
        try:
            img = Image.open(img_path).convert('RGB')
            embedding = model.encode(img, convert_to_tensor=False, normalize_embeddings=True)
            feature_list.append(embedding)
            filenames.append(img_path)
        except Exception as e:
            print(f"Error processing {file}: {e}")
# Convert and save as before
feature_list = np.array(feature_list)
pickle.dump(feature_list, open('features_clip.pkl', 'wb'))   # 👈 expected by app_streamlit.py
pickle.dump(filenames, open('filenames_clip.pkl', 'wb'))

print(f"✅ Saved {len(feature_list)} CLIP embeddings successfully as features_clip.pkl.")
