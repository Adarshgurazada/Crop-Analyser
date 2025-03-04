import os
import shutil
import random
import albumentations as A
import cv2
from tqdm import tqdm

DATASET_PATH = "data"
TRAIN_PATH = "train"  
VALIDATION_PATH = "validation"

# for oversamplig
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
])

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VALIDATION_PATH, exist_ok=True)

def move_to_validation(class_path, class_name, crop_type, keep_count=500):
    val_class_path = os.path.join(VALIDATION_PATH, crop_type, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    images = os.listdir(class_path)
    if len(images) > keep_count:
        to_move = images[keep_count:]
        for img in to_move:
            shutil.move(os.path.join(class_path, img), os.path.join(val_class_path, img))

def oversample(class_path, class_name, crop_type, target_count=600):
    train_class_path = os.path.join(TRAIN_PATH, crop_type, class_name) 
    os.makedirs(train_class_path, exist_ok=True) 

    images = os.listdir(class_path)
    if len(images) < target_count:
        while len(os.listdir(train_class_path)) < target_count: 
            img_name = random.choice(images)
            img_path = os.path.join(class_path, img_name) 

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = augment(image=image)['image']
            new_img_name = f"aug_{random.randint(10000, 99999)}.jpg"
            cv2.imwrite(os.path.join(train_class_path, new_img_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)) 
    else:
        for img_name in images:
            original_img_path = os.path.join(class_path, img_name)
            train_img_path = os.path.join(train_class_path, img_name)
            shutil.copy(original_img_path, train_img_path)


for crop in os.listdir(DATASET_PATH):
    crop_path = os.path.join(DATASET_PATH, crop)
    if os.path.isdir(crop_path):
        for disease in os.listdir(crop_path):
            class_path = os.path.join(crop_path, disease)

            if os.path.isdir(class_path):
                image_count = len(os.listdir(class_path))

                if crop == "Rice_Leaf":
                    move_to_validation(class_path, disease, crop, keep_count=500)
                    train_class_path = os.path.join(TRAIN_PATH, crop, disease)
                    os.makedirs(train_class_path, exist_ok=True)
                    for img_name in os.listdir(class_path): 
                        original_img_path = os.path.join(class_path, img_name)
                        train_img_path = os.path.join(train_class_path, img_name)
                        shutil.copy(original_img_path, train_img_path)


                else:
                    oversample(class_path, disease, crop, target_count=600)
                    move_to_validation(class_path, disease, crop, keep_count=500)
                    train_class_path = os.path.join(TRAIN_PATH, crop, disease)
                    os.makedirs(train_class_path, exist_ok=True)
                    images_to_copy = os.listdir(class_path)[:500]
                    for img_name in images_to_copy:
                        original_img_path = os.path.join(class_path, img_name)
                        train_img_path = os.path.join(train_class_path, img_name)
                        shutil.copy(original_img_path, train_img_path)


print("Dataset balancing and train/validation split complete!")
print(f"Training data saved to: {TRAIN_PATH}")
print(f"Validation data saved to: {VALIDATION_PATH}")







# import os
# import shutil
# import random
# import albumentations as A
# import cv2
# from tqdm import tqdm

# # Define paths
# DATASET_PATH = "data"
# VALIDATION_PATH = "validation"

# for oversampling
# augment = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.Rotate(limit=20, p=0.5),
# ])

# # Create validation folder
# os.makedirs(VALIDATION_PATH, exist_ok=True)

# def move_to_validation(class_path, class_name, crop_type, keep_count=500):
#     """ Move excess images to validation set """
#     val_class_path = os.path.join(VALIDATION_PATH, crop_type, class_name)
#     os.makedirs(val_class_path, exist_ok=True)

#     images = os.listdir(class_path)
#     if len(images) > keep_count:
#         to_move = images[keep_count:]
#         for img in to_move:
#             shutil.move(os.path.join(class_path, img), os.path.join(val_class_path, img))

# def oversample(class_path, class_name, crop_type, target_count=600):
#     """ Oversample minority classes using augmentation """
#     images = os.listdir(class_path)
#     if len(images) < target_count:
#         os.makedirs(class_path, exist_ok=True)
#         while len(os.listdir(class_path)) < target_count:
#             img_name = random.choice(images)
#             img_path = os.path.join(class_path, img_name)
            
#             image = cv2.imread(img_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             augmented = augment(image=image)['image']
#             new_img_name = f"aug_{random.randint(10000, 99999)}.jpg"
#             cv2.imwrite(os.path.join(class_path, new_img_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

# # Process each crop type
# for crop in os.listdir(DATASET_PATH):
#     crop_path = os.path.join(DATASET_PATH, crop)
#     if os.path.isdir(crop_path):
#         for disease in os.listdir(crop_path):
#             class_path = os.path.join(crop_path, disease)

#             if os.path.isdir(class_path):
#                 image_count = len(os.listdir(class_path))

#                 if crop == "Rice_Leaf":
#                     # Undersample Rice_Leaf to 500
#                     move_to_validation(class_path, disease, crop, keep_count=500)
                
#                 else:
#                     # Oversample others to 600
#                     oversample(class_path, disease, crop, target_count=600)
#                     move_to_validation(class_path, disease, crop, keep_count=500)

# print("Dataset balancing complete!")