import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

# ==========================================
# Configuration and Hyperparameters
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

def build_model():
    """
    Constructs the model using transfer learning. 
    MobileNetV2 acts as a frozen feature extractor.
    """
    print("Loading MobileNetV2 base model...")
    # Load MobileNetV2 with pre-trained ImageNet weights, excluding the top dense layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze the base model to prevent weights from updating during training
    base_model.trainable = False

    # Extract the output of the base model
    x = base_model.output
    
    # Add custom classification head
    x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = Dense(128, activation='relu', name='dense_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)
    
    # Binary classification logic: 1 neuron with Sigmoid activation
    predictions = Dense(1, activation='sigmoid', name='output')(x)

    # By passing base_model.input, we flatten the architecture to avoid nested model complexities.
    # This highly benefits downstream interpretability tools like Grad-CAM.
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile Model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def main():
    # Verify Directory structure
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"[Error] Please ensure your dataset exists at '{TRAIN_DIR}' and '{TEST_DIR}'")
        print("Required structure:")
        print("  dataset/train/defective/")
        print("  dataset/train/ok/")
        print("  dataset/test/defective/")
        print("  dataset/test/ok/")
        return

    print("Setting up Data Generators with Augmentation...")
    # Data Augmentation & Preprocessing for Train Data
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2  # 20% validation split from training folder
    )

    # Preprocessing for Test Data (No Augmentation)
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    # --- Generators ---
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False # Crucial for accurate classification_report mapping
    )

    print("\nBuilding model...")
    model = build_model()
    # model.summary() # Optional: print architecture summary

    # Callbacks
    # Save the model whenever validation loss improves
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    # Stop early if the model stagnates for 3 subsequent epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

    print("\nStarting Training phase...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )

    print("\nTraining completed! Evaluating on Test Set...")
    loss, accuracy, precision, recall = model.evaluate(test_generator)
    print(f"-> Test Accuracy : {accuracy*100:.2f}%")
    print(f"-> Test Precision: {precision:.4f}")
    print(f"-> Test Recall   : {recall:.4f}")
    
    # Optional F1 Score (Harmonic mean of precision and recall)
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"-> Test F1-Score : {f1_score:.4f}")

    # Generate Detailed Classification Report
    print("\nGenerating Detailed Classification Report...")
    test_generator.reset()
    preds = model.predict(test_generator)
    
    # Threshold at 0.5 for binary classes
    y_pred = (preds > 0.5).astype(int)
    y_true = test_generator.classes

    # Translate numeric indices back to original folder text outputs ('defective', 'ok')
    labels = {v: k for k, v in test_generator.class_indices.items()}
    target_names = [labels[0], labels[1]]

    print("\n" + "="*50)
    print("                CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("="*50)
    print("\nModel saved successfully as 'model.h5'.")

if __name__ == '__main__':
    main()
