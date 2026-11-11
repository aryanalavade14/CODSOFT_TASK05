import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import HandwritingCNN


# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. DEFINE PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "handwriting_cnn.pth"
)

MAPPING_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "label_mapping.json"
)


# ==========================================
# 3. LOAD CHARACTER MAPPING
# ==========================================

with open(MAPPING_PATH, "r") as file:
    label_mapping = json.load(file)

print("Character mapping loaded successfully!")


# ==========================================
# 4. LOAD TRAINED CNN MODEL
# ==========================================

model = HandwritingCNN(
    vocab_size=62,
    embedding_dim=128,
    image_size=96
).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Trained CNN model loaded successfully!")
print("Using device:", device)


# ==========================================
# 5. GENERATE SINGLE CHARACTER
# ==========================================

def generate_character(character):

    if character not in label_mapping:
        raise ValueError(
            f"Unsupported character: {character}"
        )

    character_index = label_mapping[character]

    input_tensor = torch.tensor(
        [[character_index]],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        output = model(input_tensor)

    generated_image = output.squeeze().cpu().numpy()

    return generated_image


# ==========================================
# 6. GENERATE COMPLETE TEXT
# ==========================================

def generate_text(text):

    character_images = []

    for character in text:

        if character == " ":

            blank_image = np.ones(
                (96, 30),
                dtype=np.float32
            )

            character_images.append(blank_image)

        else:

            image = generate_character(character)

            character_images.append(image)

    if not character_images:
        raise ValueError("Please enter some text.")

    final_image = np.concatenate(
        character_images,
        axis=1
    )

    return final_image


# ==========================================
# 7. DISPLAY AND SAVE GENERATED HANDWRITING
# ==========================================

def display_generated_text(text):

    generated_image = generate_text(text)

    plt.figure(
        figsize=(max(6, len(text) * 0.8), 3)
    )

    plt.imshow(
        generated_image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title("CNN Handwritten Text Generation")

    plt.axis("off")

    plt.tight_layout()

    # Save generated image
    output_path = os.path.join(
        BASE_DIR,
        "generated_handwriting.png"
    )

    plt.imsave(
        output_path,
        generated_image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    print("\nGenerated image saved at:")
    print(output_path)

    plt.show()


# ==========================================
# 8. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("   CNN HANDWRITTEN TEXT GENERATOR")
    print("================================")

    user_text = input(
        "Enter text to convert into handwriting: "
    )

    if user_text.strip():

        try:

            display_generated_text(user_text)

            print("\nHandwritten text generated successfully!")

        except ValueError as error:

            print("\nError:", error)

    else:

        print("Please enter valid text.")