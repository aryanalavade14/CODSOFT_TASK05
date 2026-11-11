import torch
import torch.nn as nn


class HandwritingCNN(nn.Module):

    def __init__(self, vocab_size=62, embedding_dim=128, image_size=96):

        super(HandwritingCNN, self).__init__()

        self.image_size = image_size

        # 1. Character Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # 2. Fully Connected Layer
        self.fc = nn.Linear(
            embedding_dim,
            256 * 6 * 6
        )

        # 3. CNN Decoder
        self.decoder = nn.Sequential(

            # Layer 1
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.ConvTranspose2d(
                32, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

        # 4. Resize Output to 96x96
        self.resize = nn.Upsample(
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):

        # Character embedding
        x = self.embedding(x)

        # Remove sequence dimension
        x = x.squeeze(1)

        # Fully connected layer
        x = self.fc(x)

        # Reshape into feature maps
        x = x.view(-1, 256, 6, 6)

        # Generate character image
        x = self.decoder(x)

        # Resize image
        x = self.resize(x)

        return x


# ==========================================
# MODEL TESTING
# ==========================================

if __name__ == "__main__":

    model = HandwritingCNN()

    sample_input = torch.tensor([
        [0],
        [1],
        [2],
        [3]
    ])

    output = model(sample_input)

    print("CNN Handwriting Model Created Successfully!")

    print("Input Shape:", sample_input.shape)

    print("Output Shape:", output.shape)

    print(
        "Total Parameters:",
        sum(p.numel() for p in model.parameters())
    )