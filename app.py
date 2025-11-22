import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract features from a single image
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img)
        expanded_arr = np.expand_dims(arr, axis=0)
        preprocessed_img = preprocess_input(expanded_arr)
        result = model.predict(preprocessed_img)
        flattened_result = result.flatten()
        normalized = flattened_result / norm(flattened_result)
        return normalized
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Folder containing your images
folder_path = "C:/Users/Abhilasha/Downloads/MCA/Python/Fashion Recommender System/images"

# Lists to store image paths and extracted features
filenames = []
feature_list = []

# Loop through all image files in the folder
for file in tqdm(os.listdir(folder_path)):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
        full_path = os.path.join(folder_path, file)
        feature = extract_features(full_path, model)
        if feature is not None:
            filenames.append(full_path)
            feature_list.append(feature)

# Save features and filenames using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print("✅ Feature extraction complete. Data saved to 'embeddings.pkl' and 'filenames.pkl'.")