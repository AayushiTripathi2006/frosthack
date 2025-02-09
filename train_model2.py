import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Sample dataset for demonstration
features = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
labels = np.array(["plastic_bottle", "can", "plastic_bottle", "can", "glass_bottle"])

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train a simple model
model = RandomForestClassifier()
model.fit(features, encoded_labels)

# Save the trained model and label encoder to model.pkl
model_data = {"model": model, "label_encoder": label_encoder}

with open("model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("Model saved successfully as model.pkl!")
