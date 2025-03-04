# Crop Disease Classification using Transfer Learning

## Overview

This repository contains code for a crop disease classification project. The project utilizes transfer learning with TensorFlow/Keras to classify crop leaves and identify diseases affecting Rice, Wheat, and Sugarcane plants.  The model employs a two-stage sequential approach:

1.  **Crop Type Classification:** First, the model identifies the type of crop leaf as Rice, Sugarcane, or Wheat.
2.  **Disease Classification:** Based on the identified crop, a specialized model classifies the specific disease affecting the leaf.

This approach allows for more accurate disease diagnosis by leveraging crop-specific disease patterns.

## Dataset

The dataset is a curated collection from publicly available Kaggle datasets, combined and structured for this project:

*   **Rice Leaf Diseases:** [Rice Diseases Image Dataset](https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset)
*   **Wheat Leaf Diseases:** [Wheat Leaf Dataset](https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset)
*   **Sugarcane Leaf Diseases:** [Sugarcane Disease Dataset](https://www.kaggle.com/datasets/prabhakaransoundar/sugarcane-disease-dataset)

The combined dataset is organized into `train` and `validation` folders, with subfolders for each crop type (`Rice_Leaf`, `Sugarcane_Leaf`, `Wheat_Leaf`). Within each crop folder, subdirectories represent the classes (crop types for the main classification, and disease types for the disease-specific models).

**Dataset Sizes:**

*   **Crop Classification:** 3 classes (Rice_Leaf, Sugarcane_Leaf, Wheat_Leaf) - 5000 training images, 1000 validation images.
*   **Rice Leaf Disease Classification:** 4 classes (BrownSpot, Healthy, Hispa, LeafBlast) - 2000 training images, 400 validation images.
*   **Sugarcane Leaf Disease Classification:** 3 classes (Bacterial_Blight, Healthy, Red_Rot) - 1500 training images, 300 validation images.
*   **Wheat Leaf Disease Classification:** 3 classes (Healthy, Septoria, Stripe_Rust) - 1500 training images, 300 validation images.

## Model Architecture

This project utilizes transfer learning with MobileNetV2 as the base architecture for all models. The model architecture is sequential and consists of two main stages:

1.  **Crop Classification Model:**
    *   **Base Model:** Pre-trained MobileNetV2 (for feature extraction).
    *   **Classification Head:** Custom layers added on top of MobileNetV2 to classify the input image into one of the three crop types (Rice, Sugarcane, Wheat).

2.  **Crop-Specific Disease Classification Models:**
    *   Three separate models are trained, one for each crop: Rice, Sugarcane, and Wheat.
    *   **Base Model:** Pre-trained MobileNetV2 (transfer learning).
    *   **Disease Classification Head:** Custom layers on top of MobileNetV2, tailored to classify diseases relevant to the specific crop.
        *   **Rice Disease Model:** Classifies 4 rice diseases.
        *   **Sugarcane Disease Model:** Classifies 3 sugarcane diseases.
        *   **Wheat Disease Model:** Classifies 3 wheat diseases.

## Training Process and Results

The models were trained using TensorFlow/Keras with the following configurations:

*   **Image Size:** 224x224 pixels
*   **Batch Size:** 32
*   **Optimizer:** Adam
*   **Loss Function:** Sparse Categorical Crossentropy
*   **Metrics:** Accuracy
*   **Callbacks:**
    *   **EarlyStopping:** Monitors validation accuracy and stops training if it plateaus, preventing overfitting and restoring the best model weights.
    *   **Custom Accuracy Callback:**  A custom callback was implemented to stop training automatically when the training accuracy exceeds 97.5% and validation accuracy exceeds 95%. This ensures efficient training by stopping once satisfactory performance is achieved.

**Training Results Summary:**

The training process yielded models with high accuracy for both crop and disease classification.

*   **Crop Classification Model:** Achieved near-perfect accuracy on the validation set within a few epochs, indicating robust crop type identification. Training stopped early due to the custom accuracy callback.
*   **Disease Classification Models (Rice, Sugarcane, Wheat):** Each crop-specific disease model demonstrated good performance in classifying diseases.
    *   The **Rice Disease Model** training was halted by the EarlyStopping callback, indicating convergence and effective learning.
    *   The **Sugarcane Disease Model** training stopped due to the custom accuracy callback, reaching the desired accuracy thresholds quickly.
    *   The **Wheat Disease Model** training followed a similar process, achieving good disease classification accuracy (detailed logs for Wheat model training are available in the training output).

Trained models are saved in the `models` directory as `.keras` files.

## Script Descriptions

This repository includes the following Python scripts:

*   **`data_dist.py`:** This script is responsible for organizing the combined dataset into the required directory structure for training. It likely takes the raw downloaded datasets and distributes the images into `train` and `validation` folders, categorized by crop type and disease. This script ensures the data is in the format expected by the `image_dataset_from_directory` function used in `train.py`.

*   **`balance.py`:**  This script likely addresses potential class imbalance issues within the datasets. Class imbalance occurs when some disease classes have significantly more images than others, which can bias the model's training. `balance.py` probably implements techniques like oversampling (duplicating images from minority classes) or undersampling (removing images from majority classes) to create a more balanced dataset for training, improving model performance, especially for less represented diseases.

*   **`model.py`:** Contains functions to define the model architectures. This file includes the `build_crop_classifier`, `build_rice_disease_classifier`, `build_sugarcane_disease_classifier`, and `build_wheat_disease_classifier` functions. These functions construct the TensorFlow/Keras models using MobileNetV2 as the base and adding custom classification heads for each task.

*   **`train.py`:** The main training script. It orchestrates the entire training process:
    1.  **Data Loading:** Calls `prepare_datasets()` to load and preprocess the training and validation data.
    2.  **Model Initialization:** Builds the crop classification and disease classification models using functions from `model.py`.
    3.  **Model Training:** Trains each model sequentially, utilizing EarlyStopping and the custom accuracy callback to optimize training.
    4.  **Model Saving:** Saves the trained models to the `models` directory.

*   **`test.py`:**  Provides a user interface to test the trained models on new images. It allows users to select an image from the `test` folder and displays the model's prediction for crop type and disease, both in the console and optionally in a GUI window (if Pillow is installed).

## How to Use the Test Script (`test.py`)

1.  **Ensure Models are Trained:** Run `train.py` to train and save the models in the `models` folder.
2.  **Prepare Test Images:** Place test images in a folder named `test` in the same directory as the script. You can organize images within subfolders inside `test`.
3.  **Install Pillow (Optional):** For visual image display with predictions, install Pillow:
    ```bash
    pip install Pillow
    ```
4.  **Run `test.py`:** Execute the test script from your terminal:
    ```bash
    python test.py
    ```
5.  **Select Image:** A file dialog will open, allowing you to select an image from the `test` folder.
6.  **View Results:**
    *   **Console Output:** The script will print the predicted crop type, disease, and disease classes to the console.
    *   **Optional GUI (if Pillow is installed):** A window will pop up displaying the selected image with the predicted crop and disease overlaid.
7.  **Test Again:** The script will ask if you want to test another image, allowing for multiple predictions without restarting.

## Dependencies

*   Python 3.x
*   TensorFlow (>= 2.x)
*   Keras
*   NumPy
*   Tkinter (usually comes with Python, for file dialog)
*   Pillow (PIL) (for optional image display in GUI)

**Installation:**

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy Pillow
    ```

## File Structure

```
Crop-Disease-Classification/
├── data/
│   ├── train/
│   │   ├── Rice_Leaf/
│   │   │   ├── BrownSpot/
│   │   │   ├── Healthy/
│   │   │   ├── Hispa/
│   │   │   └── LeafBlast/
│   │   ├── Sugarcane_Leaf/
│   │   │   ├── Bacterial_Blight/
│   │   │   ├── Healthy/
│   │   │   └── Red_Rot/
│   │   └── Wheat_Leaf/
│   │       ├── Healthy/
│   │       ├── Septoria/
│   │       └── Stripe_Rust/
│   └── validation/
│       ├── Rice_Leaf/
│       │   ├── BrownSpot/
│       │   ├── Healthy/
│       │   ├── Hispa/
│       │   └── LeafBlast/
│       ├── Sugarcane_Leaf/
│       │   ├── Bacterial_Blight/
│       │   ├── Healthy/
│       │   └── Red_Rot/
│       └── Wheat_Leaf/
│           ├── Healthy/
│           ├── Septoria/
│           └── Stripe_Rust/
├── models/
│   ├── crop_classification_model.keras
│   ├── rice_disease_classification_model.keras
│   ├── sugarcane_disease_classification_model.keras
│   └── wheat_disease_classification_model.keras
├── test/
│   ├── [user_test_images_here]
├── balance.py       # Script for data balancing (if used)
├── data_dist.py     # Script for data distribution
├── model.py         # Model building functions
├── train.py         # Training script
├── test.py    # Script for testing with user-selected images
└── README.md        # This file
```

## Model Definition (`model.py`)

The `model.py` file contains functions to build the crop and disease classification models. It defines the architecture of the MobileNetV2-based models used for both crop type and disease classification.

## Data Distribution Script (`data_dist.py`)

The `data_dist.py` script is used to preprocess and organize the dataset. It takes the raw image data and distributes it into training and validation sets, structuring the directories as required for the model training process. This script is essential for preparing the data before running the training script.

## Data Balancing Script (`balance.py`)

The `balance.py` script is designed to handle class imbalance in the dataset. It likely employs techniques to balance the number of images across different disease classes, ensuring that the model is not biased towards classes with more samples. This script is run before training to improve the fairness and overall performance of the disease classification models.

## Training Script (`train.py`)

The `train.py` script is the core script for training the models. It performs the following steps:

1.  **Data Preparation:** Loads and preprocesses the training and validation datasets using `image_dataset_from_directory`, leveraging the data organization created by `data_dist.py`.
2.  **Model Building:**  Calls functions from `model.py` to build the crop and disease classification models.
3.  **Model Training:** Trains each model sequentially, using the Adam optimizer, sparse categorical crossentropy loss, accuracy metric, and the defined EarlyStopping and custom accuracy callbacks.
4.  **Model Saving:** Saves the trained models in the `models` directory for later use in prediction.

## Testing Script (`test.py`)

The `test.py` script provides a user-friendly way to test the trained models. It allows users to select an image from the `test` folder and then uses the trained models to predict both the crop type and the disease. The results are displayed in the console, and optionally, a GUI window shows the image with the predictions.

