import os
import numpy as np
import pickle
from keras_preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from tqdm import tqdm  # progress bar

# Setup model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])
model.trainable = False

# Folder containing images
img_folder = 'images'  # change if needed

# Feature extraction function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed).flatten()
    normalized = result / norm(result)
    return normalized

# Lists to hold data
filenames = []
feature_list = []

# Loop through images and extract features
for file in tqdm(os.listdir(img_folder)):
    file_path = os.path.join(img_folder, file)
    try:
        features = extract_features(file_path, model)
        feature_list.append(features)
        filenames.append(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Save to pickle files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("✅ embeddings.pkl and filenames.pkl created successfully.")

#python -m streamlit run main.py