import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk 

MODEL_SAVE_PATH = "models"
CROP_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'crop_classification_model.keras')
RICE_DISEASE_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'rice_disease_classification_model.keras')
SUGARCANE_DISEASE_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'sugarcane_disease_classification_model.keras')
WHEAT_DISEASE_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'wheat_disease_classification_model.keras')

CROP_CLASSES = ['Rice_Leaf', 'Sugarcane_Leaf', 'Wheat_Leaf']
DISEASE_CLASSES = {
    'Rice_Leaf': ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast'],
    'Sugarcane_Leaf': ['Bacterial_Blight', 'Healthy', 'Red_Rot'],
    'Wheat_Leaf': ['Healthy', 'Septoria', 'Stripe_Rust']
}

IMG_SIZE = (224, 224) 


def load_models():
    crop_model = load_model(CROP_MODEL_PATH)
    rice_disease_model = load_model(RICE_DISEASE_MODEL_PATH)
    sugarcane_disease_model = load_model(SUGARCANE_DISEASE_MODEL_PATH)
    wheat_disease_model = load_model(WHEAT_DISEASE_MODEL_PATH)
    return crop_model, rice_disease_model, sugarcane_disease_model, wheat_disease_model


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_crop_type(img_array, model):
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    crop_type = CROP_CLASSES[predicted_class_index].split('_')[0] 
    leaf_type = CROP_CLASSES[predicted_class_index]
    return crop_type, leaf_type


def predict_disease(img_array, crop_type, rice_model, sugarcane_model, wheat_model):
    if crop_type == 'Rice':
        disease_prediction = rice_model.predict(img_array)
        disease_class_names = DISEASE_CLASSES['Rice_Leaf']
    elif crop_type == 'Sugarcane':
        disease_prediction = sugarcane_model.predict(img_array)
        disease_class_names = DISEASE_CLASSES['Sugarcane_Leaf']
    elif crop_type == 'Wheat':
        disease_prediction = wheat_model.predict(img_array)
        disease_class_names = DISEASE_CLASSES['Wheat_Leaf']
    else:
        return "Unknown Crop Type", []

    predicted_disease_index = np.argmax(disease_prediction)
    predicted_disease = disease_class_names[predicted_disease_index]
    return predicted_disease, disease_class_names


def open_file_dialog():
    root = tk.Tk()
    root.withdraw() 
    test_folder_path = "test"
    if not os.path.exists(test_folder_path) or not os.path.isdir(test_folder_path):
        print(f"Error: 'test' folder not found at '{test_folder_path}'. Please ensure the 'test' folder exists in the same directory as the script.")
        return None

    file_path = filedialog.askopenfilename(
        initialdir=test_folder_path,
        title="Select an Image from Test Folder",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    return file_path



def main():
    crop_model, rice_disease_model, sugarcane_disease_model, wheat_disease_model = load_models()

    while True: 
        image_path = open_file_dialog()
        if not image_path:
            print("No image selected. Exiting.")
            break

        try:
            img_array = preprocess_image(image_path)

            crop_type, leaf_type = predict_crop_type(img_array, crop_model)
            disease, disease_class_names = predict_disease(img_array, crop_type, rice_disease_model, sugarcane_disease_model, wheat_disease_model)


            print("\n Prediction:")
            print(f"Selected Image: {os.path.basename(image_path)}")
            print(f"Predicted Crop Type: {crop_type}")
            print(f"Predicted Disease: {disease}")
            print(f"Disease Classes for {crop_type}: {disease_class_names}") 

        except Exception as e_pred:
            print(f"Error during prediction: {e_pred}")

        another_prediction = input("Do you want to test another image? (yes/no): ").lower()
        if another_prediction != 'yes':
            break

    print("Testing finished.")


if __name__ == "__main__":
    main()