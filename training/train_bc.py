from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data.dataset import DemoDataset
from models.bc_policy import BCPolicy


def main() -> None:
    data_path = "data/demos.pkl"
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 30

    dataset = DemoDataset(data_path=data_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BCPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/bc_policy.pt")
    print("Saved model to checkpoints/bc_policy.pt")


if __name__ == "__main__":
    main()