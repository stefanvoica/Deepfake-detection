import zipfile
import os

zip_path = "deepfake-classification-unibuc.zip"
extract_dir = "."
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Folosim GPU daca este disponibil
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 125
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMAGE_SIZE = 100
NUM_CLASSES = 5

#Functie pentru incarcare imagini + augmentare
def load_dataset(df, folder, transform, augment=False):
    images, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(folder, f"{row['image_id']}.png")
        image = Image.open(path).convert("RGB")

        images.append(transform(image)) #adaugam imaginea originala
        labels.append(row['label'])

        if augment:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT) #imagine oglindita orizontal
            images.append(transform(flipped))
            labels.append(row['label'])

    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

#Calculam media si deviatia standard pentru normalizare
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0; std = 0.0; nb_samples = 0

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

#Transformare fara normalizare
transform_tmp = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()])

#Incarcam datele
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

# Calculam media si std folosind setul augmentat
train_set_tmp = load_dataset(train_df, "train", transform_tmp, augment=True)
mean, std = compute_mean_std(train_set_tmp)
print("Mean:", mean)
print("Std:", std)

# Transformari finale (cu normalizare)
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())])

transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())])

# Cream seturile de antrenare si validare
train_set = load_dataset(train_df, "train", transform_train, augment=True)  #augmentam cu flip
val_set = load_dataset(val_df, "validation", transform_val, augment=False)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

#Model MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size=3*IMAGE_SIZE*IMAGE_SIZE, num_classes=NUM_CLASSES):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),  #transformam imaginea in vector 1D
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(x)

#Initializam modelul si setarile de antrenare
model = SimpleMLP().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  #penalizare L2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()

best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_correct = train_total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(out, dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total

    #Validare
    model.eval()
    val_correct = val_total = val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            val_loss += criterion(out, y).item()
            preds = torch.argmax(out, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model.pt")  #Salvam cel mai bun model

#Functie pentru validare finala
def validate_simple(model, val_loader):
    model.eval()
    val_correct = val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = torch.argmax(out, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")

    #matricea de confuzie
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Validation Confusion Matrix")
    plt.show()

#Validam modelul antrenat
model.load_state_dict(torch.load("model.pt"))
validate_simple(model, val_loader)

#Functie pentru generarea predictiilor pe test set
def predict_on_test():
    test_df = pd.read_csv("test.csv")
    test_ids, test_images = [], []

    for _, row in test_df.iterrows():
        path = os.path.join("test", f"{row['image_id']}.png")
        image = Image.open(path).convert("RGB")
        test_images.append(transform_val(image))
        test_ids.append(row['image_id'])

    model = SimpleMLP().to(DEVICE)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_images), BATCH_SIZE):
            batch = torch.stack(test_images[i:i + BATCH_SIZE]).to(DEVICE)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            predictions.extend(preds)

    submission_df = pd.DataFrame({
        "image_id": test_ids,
        "label": predictions
    })
    submission_df.to_csv("submission_mlp.csv", index=False)
    print("Predictii test salvate!")

predict_on_test()