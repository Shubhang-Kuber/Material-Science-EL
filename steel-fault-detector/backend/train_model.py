"""
CNN Model Training Script for Steel Plate Fault Detection
This script creates and trains a CNN model for classifying steel plate faults from images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 6  # NEU-DET has 6 defect classes

# NEU-DET Dataset Classes
FAULT_TYPES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]

# Mapping for display names
FAULT_DISPLAY_NAMES = {
    'crazing': 'Crazing',
    'inclusion': 'Inclusion',
    'patches': 'Patches',
    'pitted_surface': 'Pitted Surface',
    'rolled-in_scale': 'Rolled-in Scale',
    'scratches': 'Scratches'
}

def create_cnn_model_from_scratch():
    """
    Create a custom CNN model from scratch for steel fault detection
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fifth Convolutional Block
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        
        # Dense Layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer (7 fault types)
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_transfer_learning_model():
    """
    Create a CNN model using transfer learning with MobileNetV2
    Better for limited training data
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_data_generators(train_dir, val_dir=None):
    """
    Create data generators for training and validation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2 if val_dir is None else 0.0
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    if val_dir is None:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator, model_name='steel_fault_cnn'):
    """
    Train the CNN model
    """
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/{model_name}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"📊 Training history saved to {save_path}")

def create_sample_model():
    """
    Create and save a sample model without training data
    (For testing purposes)
    """
    print("🔧 Creating sample CNN model...")
    
    model = create_cnn_model_from_scratch()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save('models/steel_fault_cnn.h5')
    print("✅ Sample model saved to models/steel_fault_cnn.h5")
    print("⚠️ Note: This is an untrained model. Train with real data for accurate predictions.")
    
    return model

def main():
    """
    Main training script using NEU-DET dataset
    """
    print("=" * 60)
    print("🏭 Steel Plate Fault Detection - CNN Model Training")
    print("    Using NEU-DET Dataset")
    print("=" * 60)
    print(f"\n📊 Defect Classes: {FAULT_TYPES}")
    print(f"📐 Image Size: {IMG_SIZE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔄 Max Epochs: {EPOCHS}")
    print(f"🎯 Number of Classes: {NUM_CLASSES}")
    
    # NEU-DET dataset paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    neu_det_path = os.path.join(os.path.dirname(base_path), 'NEU-DET')
    
    train_dir = os.path.join(neu_det_path, 'train', 'images')
    val_dir = os.path.join(neu_det_path, 'validation', 'images')
    
    print(f"\n📁 Dataset path: {neu_det_path}")
    print(f"📁 Train directory: {train_dir}")
    print(f"📁 Validation directory: {val_dir}")
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"\n✅ Found NEU-DET dataset!")
        
        # Count images per class
        print("\n📈 Dataset Statistics:")
        total_train = 0
        total_val = 0
        for class_name in FAULT_TYPES:
            train_class_path = os.path.join(train_dir, class_name)
            val_class_path = os.path.join(val_dir, class_name)
            
            train_count = len(os.listdir(train_class_path)) if os.path.exists(train_class_path) else 0
            val_count = len(os.listdir(val_class_path)) if os.path.exists(val_class_path) else 0
            total_train += train_count
            total_val += val_count
            
            print(f"   {class_name}: {train_count} train, {val_count} validation")
        
        print(f"\n   Total: {total_train} train images, {total_val} validation images")
        
        # Create data generators
        print("\n🔧 Creating data generators...")
        train_gen, val_gen = create_data_generators(train_dir, val_dir)
        
        print(f"\n📊 Classes found: {train_gen.class_indices}")
        
        # Create model with Transfer Learning
        print("\n🤖 Creating CNN model with Transfer Learning (MobileNetV2)...")
        model = create_transfer_learning_model()
        
        # Print model summary
        print("\n📋 Model Architecture:")
        model.summary()
        
        # Train the model
        history = train_model(model, train_gen, val_gen)
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate final model
        print("\n📊 Final Evaluation:")
        val_loss, val_acc = model.evaluate(val_gen)
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"📁 Model saved to: models/steel_fault_cnn.h5")
        print("=" * 60)
        
    else:
        print(f"\n❌ NEU-DET dataset not found!")
        print(f"   Expected train path: {train_dir}")
        print(f"   Expected val path: {val_dir}")
        print("\n🔧 Creating sample untrained model for testing...")
        create_sample_model()

if __name__ == '__main__':
    main()
