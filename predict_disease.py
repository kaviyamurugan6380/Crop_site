from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = load_model('models/plant_disease_model.h5')

# Get class labels
class_names = os.listdir('dataset/train')
class_names.sort()   

# Load test image
img_path=input("Enter image path:")
img=image.load_img(img_path,target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)/255

# Predict
prediction = model.predict(img_array)

# Show result
print("Disease:", class_names[np.argmax(prediction)])