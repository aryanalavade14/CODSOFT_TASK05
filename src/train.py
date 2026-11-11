import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from model import HandwritingCNN


# ==========================================
# CONFIGURATION
# ==========================================

EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "processed_dataset.npz"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# CUSTOM DATASET
# ==========================================

class HandwritingDataset(Dataset):

    def __init__(self, images, labels):

        self.images = torch.tensor(
            images,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return self.labels[index].unsqueeze(0), self.images[index]


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

print("\nLoading processed dataset...")

data = np.load(DATASET_PATH)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)


# ==========================================
# DATA LOADERS
# ==========================================

train_dataset = HandwritingDataset(X_train, y_train)
test_dataset = HandwritingDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# INITIALIZE MODEL
# ==========================================

model = HandwritingCNN(
    vocab_size=62,
    embedding_dim=128,
    image_size=96
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nModel initialized successfully!")
print("Training device:", DEVICE)


# ==========================================
# TRAINING
# ==========================================

train_losses = []
test_losses = []

print("\nStarting CNN Model Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_train_loss = 0.0

    for labels, images in train_loader:

        labels = labels.to(DEVICE)
        images = images.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(labels)

        loss = criterion(outputs, images)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)

    train_losses.append(epoch_train_loss)

    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    running_test_loss = 0.0

    with torch.no_grad():

        for labels, images in test_loader:

            labels = labels.to(DEVICE)
            images = images.to(DEVICE)

            outputs = model(labels)

            loss = criterion(outputs, images)

            running_test_loss += loss.item()

    epoch_test_loss = running_test_loss / len(test_loader)

    test_losses.append(epoch_test_loss)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {epoch_train_loss:.5f} "
        f"Test Loss: {epoch_test_loss:.5f}"
    )


# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "handwriting_cnn.pth"
)

torch.save(model.state_dict(), model_path)

print("\nCNN Model Saved Successfully!")
print("Model location:", model_path)


# ==========================================
# SAVE TRAINING GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Testing Loss")

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")

plt.title("CNN Handwriting Model Training")

plt.legend()
plt.grid(True)

plt.tight_layout()

graph_path = os.path.join(
    MODEL_DIR,
    "cnn_training_loss.png"
)

plt.savefig(graph_path)

print("Training graph saved at:", graph_path)

plt.show()

print("\nCNN TRAINING COMPLETED!")