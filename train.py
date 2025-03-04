import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, Callback 
import os
from model import build_crop_classifier, build_rice_disease_classifier, build_sugarcane_disease_classifier, build_wheat_disease_classifier

TRAIN_PATH = "train"
VALIDATION_PATH = "validation"
MODEL_SAVE_PATH = "models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


IMG_SIZE = 224
BATCH_SIZE = 32


CROP_CLASSES = ['Rice_Leaf', 'Sugarcane_Leaf', 'Wheat_Leaf']
DISEASE_CLASSES = {
    'Rice_Leaf': ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast'],
    'Sugarcane_Leaf': ['Bacterial_Blight', 'Healthy', 'Red_Rot'],
    'Wheat_Leaf': ['Healthy', 'Septoria', 'Stripe_Rust']
}



class AccValCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if train_acc is not None and val_acc is not None:
            if train_acc > 0.975 and val_acc > 0.95:
                print(f"\nReached train_accuracy > 0.975 and val_accuracy > 0.95. Stopping training!")
                self.model.stop_training = True


def prepare_datasets():
    train_dataset = image_dataset_from_directory(
        TRAIN_PATH,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_names=CROP_CLASSES
    )

    validation_dataset = image_dataset_from_directory(
        VALIDATION_PATH,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_names=CROP_CLASSES 
    )

   
    rice_train_dataset = train_dataset.filter(lambda img, labels: tf.reduce_any(tf.equal(labels, 0))) # 0: Rice
    rice_validation_dataset = validation_dataset.filter(lambda img, labels: tf.reduce_any(tf.equal(labels, 0)))

    sugarcane_train_dataset = train_dataset.filter(lambda img, labels: tf.reduce_any(tf.equal(labels, 1))) # 1: Sugarcane
    sugarcane_validation_dataset = validation_dataset.filter(lambda img, labels: tf.reduce_any(tf.equal(labels, 1)))

    wheat_train_dataset = train_dataset.filter(lambda img, labels: tf.reduce_any(tf.equal(labels, 2))) # 2: Wheat
    wheat_validation_dataset = validation_dataset.filter(lambda img, labels: tf.reduce_any(tf.equal(labels, 2)))


    rice_train_dataset_disease = prepare_disease_dataset(TRAIN_PATH, 'Rice_Leaf', DISEASE_CLASSES['Rice_Leaf'])
    rice_validation_dataset_disease = prepare_disease_dataset(VALIDATION_PATH, 'Rice_Leaf', DISEASE_CLASSES['Rice_Leaf'])

    sugarcane_train_dataset_disease = prepare_disease_dataset(TRAIN_PATH, 'Sugarcane_Leaf', DISEASE_CLASSES['Sugarcane_Leaf'])
    sugarcane_validation_dataset_disease = prepare_disease_dataset(VALIDATION_PATH, 'Sugarcane_Leaf', DISEASE_CLASSES['Sugarcane_Leaf'])

    wheat_train_dataset_disease = prepare_disease_dataset(TRAIN_PATH, 'Wheat_Leaf', DISEASE_CLASSES['Wheat_Leaf'])
    wheat_validation_dataset_disease = prepare_disease_dataset(VALIDATION_PATH, 'Wheat_Leaf', DISEASE_CLASSES['Wheat_Leaf'])


    return train_dataset, validation_dataset, rice_train_dataset_disease, rice_validation_dataset_disease, sugarcane_train_dataset_disease, sugarcane_validation_dataset_disease, wheat_train_dataset_disease, wheat_validation_dataset_disease


def prepare_disease_dataset(data_dir, crop_name, disease_classes):
    """Prepares a disease dataset for a specific crop."""
    crop_path = os.path.join(data_dir, crop_name)
    dataset = image_dataset_from_directory(
        crop_path,
        labels='inferred',
        label_mode='int',
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_names=disease_classes 
    )
    return dataset



def train_crop_model(train_dataset, validation_dataset):
    print("Training Crop Classification Model...")
    crop_model = build_crop_classifier()

    early_stopping_crop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
    acc_val_callback_crop = AccValCallback()

    crop_history = crop_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=50,
        callbacks=[early_stopping_crop, acc_val_callback_crop]
    )

    crop_model.save(os.path.join(MODEL_SAVE_PATH, 'crop_classification_model.keras'))
    print("Crop Classification Model Training Complete and Saved.")
    return crop_model, crop_history



def train_rice_disease_model(train_dataset, validation_dataset):
    print("Training Rice Disease Classification Model...")
    rice_disease_model = build_rice_disease_classifier()

    early_stopping_rice_disease = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, restore_best_weights=True)
    acc_val_callback_rice_disease = AccValCallback() 

    rice_disease_history = rice_disease_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=50,
        callbacks=[early_stopping_rice_disease, acc_val_callback_rice_disease] 
    )

    rice_disease_model.save(os.path.join(MODEL_SAVE_PATH, 'rice_disease_classification_model.keras'))
    print("Rice Disease Classification Model Training Complete and Saved.")
    return rice_disease_model, rice_disease_history



def train_sugarcane_disease_model(train_dataset, validation_dataset):
    print("Training Sugarcane Disease Classification Model...")
    sugarcane_disease_model = build_sugarcane_disease_classifier()

    early_stopping_sugarcane_disease = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, restore_best_weights=True)
    acc_val_callback_sugarcane_disease = AccValCallback() 

    sugarcane_disease_history = sugarcane_disease_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=50,
        callbacks=[early_stopping_sugarcane_disease, acc_val_callback_sugarcane_disease] 
    )

    sugarcane_disease_model.save(os.path.join(MODEL_SAVE_PATH, 'sugarcane_disease_classification_model.keras'))
    print("Sugarcane Disease Classification Model Training Complete and Saved.")
    return sugarcane_disease_model, sugarcane_disease_history



def train_wheat_disease_model(train_dataset, validation_dataset):
    print("Training Wheat Disease Classification Model...")
    wheat_disease_model = build_wheat_disease_classifier()

    early_stopping_wheat_disease = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, restore_best_weights=True)
    acc_val_callback_wheat_disease = AccValCallback() 

    wheat_disease_history = wheat_disease_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=50,
        callbacks=[early_stopping_wheat_disease, acc_val_callback_wheat_disease]
    )

    wheat_disease_model.save(os.path.join(MODEL_SAVE_PATH, 'wheat_disease_classification_model.keras'))
    print("Wheat Disease Classification Model Training Complete and Saved.")
    return wheat_disease_model, wheat_disease_history



if __name__ == "__main__":
    (crop_train_data, crop_val_data,
     rice_train_data_disease, rice_val_data_disease,
     sugarcane_train_data_disease, sugarcane_val_data_disease,
     wheat_train_data_disease, wheat_validation_dataset_disease) = prepare_datasets()

    crop_classification_model, crop_training_history = train_crop_model(crop_train_data, crop_val_data)
    rice_disease_classification_model, rice_disease_training_history = train_rice_disease_model(rice_train_data_disease, rice_val_data_disease)
    sugarcane_disease_classification_model, sugarcane_disease_training_history = train_sugarcane_disease_model(sugarcane_train_data_disease, sugarcane_val_data_disease)
    wheat_disease_classification_model, wheat_disease_training_history = train_wheat_disease_model(wheat_train_data_disease, wheat_validation_dataset_disease)


    print("Models Training Completed.")