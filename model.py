import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0

IMG_SIZE = (224, 224, 3)

def build_crop_classifier():
    """Stage 1: Classifies crop type (Rice, Sugarcane, Wheat)"""

    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
    base_model.trainable = True

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_rice_disease_classifier():
    """Stage 2a: Disease classifier for Rice crop"""

    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
    base_model.trainable = True

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_sugarcane_disease_classifier():
    """Stage 2b: Disease classifier for Sugarcane crop"""

    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
    base_model.trainable = True

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_wheat_disease_classifier():
    """Stage 2c: Disease classifier for Wheat crop"""

    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
    base_model.trainable = True

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    crop_model = build_crop_classifier()
    rice_disease_model = build_rice_disease_classifier()
    sugarcane_disease_model = build_sugarcane_disease_classifier()
    wheat_disease_model = build_wheat_disease_classifier()

    print("Crop Classification Model Summary:")
    crop_model.summary()
    print("\nRice Disease Classification Model Summary:")
    rice_disease_model.summary()
    print("\nSugarcane Disease Classification Model Summary:")
    sugarcane_disease_model.summary()
    print("\nWheat Disease Classification Model Summary:")
    wheat_disease_model.summary()