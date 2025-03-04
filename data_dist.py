import os
import matplotlib.pyplot as plt

DATASET_PATH = "data"

data_distribution = {}
for crop in os.listdir(DATASET_PATH):
    crop_path = os.path.join(DATASET_PATH, crop)
    
    if os.path.isdir(crop_path):  
        for disease in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease)
            
            if os.path.isdir(disease_path): 
                image_count = len(os.listdir(disease_path))
                data_distribution[f"{crop} - {disease}"] = image_count

data_distribution = dict(sorted(data_distribution.items(), key=lambda item: item[1], reverse=True))

for category, count in data_distribution.items():
    print(f"{category}: {count} images")


# Plot
plt.figure(figsize=(12, 6))
plt.bar(data_distribution.keys(), data_distribution.values(), color='skyblue')
plt.xticks(rotation=90)
plt.xlabel("Class (Crop - Disease)")
plt.ylabel("Number of Images")
plt.title("Dataset Distribution")
plt.show()
