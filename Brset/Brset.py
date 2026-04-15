import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import numpy as np


CSV_PATH = "Business - Documents/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/labels_brset.csv"
IMG_DIR = Path("Business - Documents/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/fundus_photos")

BATCH_SIZE = 32

DISEASE_COLS = [
    "diabetic_retinopathy",
    "macular_edema",
    "amd",
    "hypertensive_retinopathy",
    "hemorrhage",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

class BRSETDataset(Dataset):
    def __init__(self, df, img_dir, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row.iloc[0]
        img_path = self.img_dir / f"{img_id}.jpg"

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype("float32"))
        return image, labels

class BRSETResNet(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_labels)

    def forward(self, x):
        logits = self.backbone(x)
        return logits   


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        probs = torch.sigmoid(logits)

        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            loss = criterion(probs, labels)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    aurocs = {}
    for i, name in enumerate(DISEASE_COLS):
        y_true = all_labels[:, i]
        y_score = all_probs[:, i]
        if len(np.unique(y_true)) < 2:
            aurocs[name] = np.nan
            continue
        aurocs[name] = roc_auc_score(y_true, y_score)

    avg_auroc = np.nanmean(list(aurocs.values()))
    return running_loss / len(loader.dataset), avg_auroc, aurocs


def train_model(train_loader, val_loader, num_labels, device, num_epochs=10):
    model = BRSETResNet(num_labels).to(device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_auroc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_avg_auroc, val_aurocs = eval_epoch(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val AUROC (avg): {val_avg_auroc:.4f}")
        for name, auc in val_aurocs.items():
            print(f"    {name}: {auc:.4f}" if not np.isnan(auc) else f"    {name}: NA")

        if val_avg_auroc > best_val_auroc:
            best_val_auroc = val_avg_auroc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest val AUROC: {best_val_auroc:.4f}")
    return model


def test_model(model, test_loader, device):
    criterion = nn.BCELoss()
    test_loss, test_avg_auroc, test_aurocs = eval_epoch(model, test_loader, criterion, device)

    print("\n=== TEST RESULTS ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test AUROC (avg): {test_avg_auroc:.4f}")
    for name, auc in test_aurocs.items():
        print(f"  {name}: {auc:.4f}" if not np.isnan(auc) else f"  {name}: NA")


def main():
    labels_df = pd.read_csv(CSV_PATH)

    print("Selected label columns:", DISEASE_COLS)
    print("Class counts:\n", labels_df[DISEASE_COLS].sum())

    label_sums = labels_df[DISEASE_COLS].sum(axis=1)

    train_df, temp_df = train_test_split(
        labels_df,
        test_size=0.3,
        random_state=42,
        stratify=label_sums
    )

    temp_label_sums = temp_df[DISEASE_COLS].sum(axis=1)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_label_sums
    )

    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = BRSETDataset(train_df, IMG_DIR, DISEASE_COLS, transform=train_transform)
    val_ds   = BRSETDataset(val_df,   IMG_DIR, DISEASE_COLS, transform=eval_transform)
    test_ds  = BRSETDataset(test_df,  IMG_DIR, DISEASE_COLS, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)

    model = train_model(train_loader, val_loader, num_labels=len(DISEASE_COLS), device=DEVICE, num_epochs=10)
    test_model(model, test_loader, DEVICE)


if __name__ == "__main__":
    main()