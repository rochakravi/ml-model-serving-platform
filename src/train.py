import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import SimpleClassifier

PROCESSED_DIR = "data/processed"

def load_training_data():
    X = []
    y = []

    files = os.listdir(PROCESSED_DIR)

    for index, file in enumerate(files):
        path = os.path.join(PROCESSED_DIR, file)

        image = np.load(path)
        print("File:", file)
        print("Shape:", image.shape)

        X.append(image)

        # dummy label
        label = index % 2
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = torch.tensor(X).float()
    y = torch.tensor(y).long()

    return X, y


def train():
    X, y = load_training_data()

    model = SimpleClassifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "model.pth")

    print("Model saved!")


if __name__ == "__main__":
    train()