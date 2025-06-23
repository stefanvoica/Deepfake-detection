import zipfile
import os

zip_path = "deepfake-classification-unibuc.zip"
extract_dir = "."

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Folosim GPU pentru antrenare daca este disponibil
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constante de antrenare
EPOCHS = 125 #nr de epoci
BATCH_SIZE = 32 #dimensiunea unui batch
LEARNING_RATE = 1e-3 #rata de invatare
IMAGE_SIZE = 100
NUM_CLASSES = 5

#Prima retea CNN
class CNN128(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN128, self).__init__()
        self.features = nn.Sequential(
            #3*100*100 => 32*100*100 => 32*50*50
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            #32*50*50 =>  64*50*50 => 64*25*25
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            #64*25*25 => 128*25*25 => 128*12*12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            #128*12*12 => 256*12*12 => 256*6*6
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.features(x) #aplicam filtrele conv
        x = self.classifier(x)#aplicam straturile fc
        return x

#al doilea CNN, mai adanc
class CNN256(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        #Singura diferenta consta in marirea nr de neuroni de pe ultimul hidden layer
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#functie pt incarcarea datelor
#argumentul augment este folosit pt a oglindi doar imaginile pe setul de training
def load_dataset(df, folder, transform, augment=False):
    images, labels = [], []
    for _, row in df.iterrows():
        path = os.path.join(folder, f"{row['image_id']}.png")
        img = Image.open(path).convert("RGB")

        images.append(transform(img)) #adaugam imaginea originala
        labels.append(row['label']) #cu labelul ei

        if augment: #daca vrem sa augmentam
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT) #oglindim imaginea
            images.append(transform(img_flip)) #si o adaugam
            labels.append(row['label'])

    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))

#calculul mediei si al deviatiei standard
def compute_mean_std(dataset):
    #cu DataLoader putem itera prin batch-uri
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0; std = 0.0; total = 0

    for data, _ in loader:
        b = data.size(0) #nr de imagini din batch-ul curent
        #(batch, channel, H*W)
        data = data.view(b, data.size(1), -1) #-1 = calc automat dim
        mean += data.mean(2).sum(0) #media pe fiecare canal si apoi suma pe integul batch
        std += data.std(2).sum(0)
        total += b

    mean /= total
    std /= total
    return mean, std

# transformari initiale pt calcule
transform_tmp = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),transforms.ToTensor()])

#Incarcarea datelor din CSV
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
#Incarcam datele de training
train_set_tmp = load_dataset(train_df, "train", transform_tmp, augment=True)
mean, std = compute_mean_std(train_set_tmp) #Compunem media si std
print("Mean:", mean)
print("Std:", std)

#Nicio augmentare suplimentara pe codul 1
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist())])

#Transformari pe validation
#Ramane la fel pe tot parcursul codului
transform_val = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist())])

#Incarcam seturile de date cu augmentare doar la training
train_set = load_dataset(train_df, "train", transform_train, augment=True)
val_set = load_dataset(val_df, "validation", transform_val, augment=False)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

#Folosim modelul CNN256
model = CNN256().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) #weight_decay - penalizeaza greutati prea mari
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    train_correct = train_total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE) #mutam batch-ul actual pe GPU
        optimizer.zero_grad() #resteam gradientii
        out = model(x) #forward
        loss = criterion(out, y) #aplicarea functiei de loss
        loss.backward() #backpropagation
        optimizer.step() #Updatam weight-urile
        #Alegem clasa cu scorul cel mai mare
        preds = torch.argmax(out, dim=1)
        train_correct += (preds == y).sum().item() #numaram cate predictii corecte
        train_total += y.size(0) #totalul imaginilor

    train_acc = train_correct / train_total #acuratetea pe training

    #Evaluam asemenator pentru validare
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

    val_acc = val_correct / val_total #acuratetea pe validare
    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    scheduler.step() #updatam LR dupa scheduler

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model1.pt") #Salvam cel mai bun model

import torch.optim as optim
#Si in acest model nu o adaug augmentari suplimentare
#Dar voi folosi modelul CNN128
model = CNN128().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
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
        torch.save(model.state_dict(), "model2.pt")

#Asemenator cu model1
model = CNN256().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4) #modificam weight_decay la 0.0005
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
        torch.save(model.state_dict(), "model3.pt")

#In acest model, am folosim o augmentare suplimentara
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=5), #Rotatia imaginii cu o val aleatoare intre -5 si 5 grade
    transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist())])


#Revenim la weight_decay 0.0001
model = CNN256().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
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
        torch.save(model.state_dict(), "model4.pt")

#Incarcam modelele antrenate deja
model_paths = [("model1.pt", CNN256), ("model2.pt", CNN128), ("model3.pt", CNN256), ("model4.pt", CNN256)]
transform = transform_val
models = []
for path, cls in model_paths:
    m = cls().to(DEVICE)  #Initializam modelul pe GPU
    m.load_state_dict(torch.load(path, map_location=DEVICE))  #Incarcam weight-urile salvate
    m.eval()
    models.append(m)  #Adaugam in lista modelele in stare de evaluare

#Functie de validare a ansamblului de modele
def validare_ensemble():
    #Incarcam CSV-ul
    val_df = pd.read_csv("validation.csv")
    val_images, val_labels = [], []
    #Incarcam fiecare imagine si aplicam transformarile
    for _, row in val_df.iterrows():
        path = os.path.join("validation", f"{row['image_id']}.png")
        img = Image.open(path).convert("RGB")
        val_images.append(transform(img))
        val_labels.append(row['label'])

    val_labels = torch.tensor(val_labels, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(torch.stack(val_images), val_labels), batch_size=BATCH_SIZE)

    toate_predictiile = []

    with torch.no_grad():  #Nu calculam gradienti
        for x, _ in val_loader:
            x = x.to(DEVICE)
            #Predictii softmax de la fiecare model
            outputs = [torch.softmax(m(x), dim=1) for m in models]
            #Facem media predictiilor
            media = sum(outputs) / len(models)
            #Alegem clasa cu scorul cel mai mare
            preds = torch.argmax(media, dim=1).cpu()
            toate_predictiile.extend(preds.tolist())

    #Acuratetea
    corecte = sum([p == t for p, t in zip(toate_predictiile, val_labels)])
    acc = corecte / len(val_labels)
    print(f"Acuratate validare ansamblu: {acc:.4f}")

    #Afisam matricea de confuzie
    cm = confusion_matrix(val_labels, toate_predictiile)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Matrice Confuzie - Validare Ansamblu")
    plt.show()


#Functie pentru testare
def test_ensemble():
    #Incarcam fisierul CSV cu imaginile
    test_df = pd.read_csv("test.csv")
    test_ids, test_images = [], []
    for _, row in test_df.iterrows():
        path = os.path.join("test", f"{row['image_id']}.png")
        image = Image.open(path).convert("RGB")
        test_images.append(transform(image))
        test_ids.append(row['image_id'])

    predictions = []
    with torch.no_grad():  #Fara gradienti atat la test, cat si la validare
        for i in range(0, len(test_images), BATCH_SIZE):
            batch = torch.stack(test_images[i:i + BATCH_SIZE]).to(DEVICE)
            outputs = [torch.softmax(model(batch), dim=1) for model in models] #Predictii softmax pe batch
            avg_output = sum(outputs) / len(models) #Facem media scorurilor
            preds = torch.argmax(avg_output, dim=1).cpu().tolist() #Clasa cu scorul cel mai mare
            predictions.extend(preds)

    #Salvam predictiile in CSV
    submission_df = pd.DataFrame({
        "image_id": test_ids,
        "label": predictions})
    submission_df.to_csv("submission_ensemble_simplu.csv", index=False)
    print("Predictii salvate cu succes!")

validare_ensemble()
test_ensemble()