import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define dataset paths
train_dir = r'C:\Users\yuvra\Documents\pp\train'  # Path to training data
test_dir = r'C:\Users\yuvra\Documents\pp\test'    # Path to testing data
model_save_path = r'C:\Users\yuvra\Documents\mlname.keras'

# Check if dataset folders exist
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError(f"Train or Test dataset not found. Please check the paths: \nTrain: {train_dir}\nTest: {test_dir}")

# Data augmentation and preprocessing
data_gen = ImageDataGenerator(rescale=1.0/255.0)

train_data = data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,  # Increased batch size for faster training
    class_mode='categorical'
)

test_data = data_gen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,  # Keeping batch size same for consistency
    class_mode='categorical'
)

# Dynamically set steps per epoch based on dataset size
steps_per_epoch = len(train_data)  # Total training batches
validation_steps = len(test_data)  # Total validation batches

# Define custom CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),  # Patience increased for large dataset
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6)  # LR decreases after 4 epochs of no improvement
]

# Train model
try:
    model.fit(
        train_data,
        validation_data=test_data,
        epochs=50,  # Increased epochs for better learning
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    # Save trained model
    model.save(model_save_path)
    print(f"Model successfully saved at {model_save_path}")
except Exception as e:
    print(f"Error during training: {e}")
