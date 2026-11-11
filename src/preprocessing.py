
import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ==========================================
# 1. DEFINE DATASET PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "dataset", "english.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

IMAGE_SIZE = 96

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():

    df = pd.read_csv(CSV_PATH)

    print("\n========== DATASET INFORMATION ==========")

    print("Dataset loaded successfully!")
    print("Total images:", len(df))
    print("Total unique characters:", df["label"].nunique())

    print("\nCharacter distribution:")
    print(df["label"].value_counts().sort_index())

    return df


# ==========================================
# 3. DISPLAY RANDOM SAMPLES
# ==========================================

def display_samples(df, number=10):

    samples = df.sample(n=number, random_state=42)

    plt.figure(figsize=(15, 6))

    for i, (_, row) in enumerate(samples.iterrows()):

        image_path = os.path.join(BASE_DIR, "dataset", row["image"])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        plt.subplot(2, 5, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {row['label']}")
        plt.axis("off")

    plt.suptitle("Original Handwritten Character Samples")
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. IMPROVED IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path, size=IMAGE_SIZE):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    # Detect dark character pixels
    _, threshold = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find character boundaries
    coordinates = cv2.findNonZero(threshold)

    if coordinates is not None:

        x, y, w, h = cv2.boundingRect(coordinates)

        # Crop unnecessary margins
        image = image[y:y+h, x:x+w]

    # Preserve aspect ratio
    h, w = image.shape

    scale = (size * 0.75) / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create white background canvas
    canvas = np.ones((size, size), dtype=np.uint8) * 255

    # Center the character
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = image

    # Normalize pixel values
    image = canvas.astype(np.float32) / 255.0

    return image


# ==========================================
# 5. PREPARE DATASET
# ==========================================

def prepare_dataset(df):

    images = []
    labels = []

    print("\n========== IMAGE PREPROCESSING ==========")

    for index, row in df.iterrows():

        image_path = os.path.join(BASE_DIR, "dataset", row["image"])

        image = preprocess_image(image_path)

        images.append(image)
        labels.append(row["label"])

        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1}/{len(df)} images")

    images = np.array(images, dtype=np.float32)

    labels = np.array(labels)

    # PyTorch channel dimension
    images = np.expand_dims(images, axis=1)

    print("\nPreprocessing completed successfully!")

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    print("Minimum pixel value:", images.min())
    print("Maximum pixel value:", images.max())

    return images, labels


# ==========================================
# 6. ENCODE CHARACTER LABELS
# ==========================================

def encode_labels(labels):

    unique_labels = sorted(set(labels))

    label_mapping = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    encoded_labels = np.array(
        [label_mapping[label] for label in labels],
        dtype=np.int64
    )

    mapping_path = os.path.join(
        PROCESSED_DIR,
        "label_mapping.json"
    )

    with open(mapping_path, "w") as file:
        json.dump(label_mapping, file, indent=4)

    print("\nCharacter encoding completed!")
    print("Total classes:", len(label_mapping))

    return encoded_labels, label_mapping


# ==========================================
# 7. SPLIT DATASET
# ==========================================

def split_dataset(images, labels):

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    print("\n========== DATASET SPLITTING ==========")

    print("Training images:", X_train.shape)
    print("Testing images:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================
# 8. SAVE PROCESSED DATASET
# ==========================================

def save_processed_data(X_train, X_test, y_train, y_test):

    save_path = os.path.join(
        PROCESSED_DIR,
        "processed_dataset.npz"
    )

    np.savez_compressed(
        save_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    print("\nProcessed dataset saved successfully!")
    print("Saved at:", save_path)


# ==========================================
# 9. MAIN FUNCTION
# ==========================================

if __name__ == "__main__":

    df = load_dataset()

    display_samples(df)

    images, labels = prepare_dataset(df)

    encoded_labels, label_mapping = encode_labels(labels)

    X_train, X_test, y_train, y_test = split_dataset(
        images, encoded_labels
    )

    save_processed_data(
        X_train, X_test, y_train, y_test
    )

    print("\n====================================")
    print("IMPROVED DATA PREPROCESSING COMPLETED!")
    print("====================================")
