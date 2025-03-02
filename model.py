import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import string

# Wczytanie danych
df = pd.read_csv("hate_speech_dataset.csv")

import pandas as pd
import re

# Wczytaj dane z BAN-PL.csv
ban_pl_df = pd.read_csv("BAN-PL.csv")
# Wybierz tylko kolumny Text i Class i utwórz jawną kopię
ban_pl_data = ban_pl_df[["Text", "Class"]].copy()
# Zmień nazwy kolumn na odpowiadające hate_speech_dataset.csv
ban_pl_data.columns = ["text", "label"]

# Wczytaj dane z BAN-PL2.csv
ban_pl2_df = pd.read_csv("BAN-PL2.csv")
# Wybierz tylko kolumny Text i Class i utwórz jawną kopię
ban_pl2_data = ban_pl2_df[["Text", "Class"]].copy()
# Zmień nazwy kolumn na odpowiadające hate_speech_dataset.csv
ban_pl2_data.columns = ["text", "label"]

def clean_text(text):
    """Funkcja do czyszczenia tekstu z niepotrzebnych znaków i formatowań"""
    # Konwertuj wartości nan/None/float na pusty string
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Usuń wszystkie wzmianki użytkowników (np. {USERNAME})
    text = re.sub(r"\{USERNAME\}", "", text)
    # Usuń wzmianki nazwisk (np. [surname])
    text = re.sub(r"\[surname\]", "", text)
    # Usuń URL-e
    text = re.sub(r"\{URL\}", "", text)
    # Usuń inne specjalne formatowania
    text = re.sub(r"\[.*?\]", "", text)
    # Usuń zbędne białe znaki
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Zastosuj funkcję czyszczącą na kolumnie text
ban_pl_data["text"] = ban_pl_data["text"].apply(clean_text)
ban_pl2_data["text"] = ban_pl2_data["text"].apply(clean_text)

# Upewnij się, że wszystkie wartości w kolumnie tekst są stringami
ban_pl_data["text"] = ban_pl_data["text"].astype(str)
ban_pl2_data["text"] = ban_pl2_data["text"].astype(str)

# Połącz dane z obu plików BAN
combined_df = pd.concat([ban_pl_data, ban_pl2_data], ignore_index=True)

# Usuń puste wiersze (te, w których tekst jest pusty)
combined_df = combined_df[combined_df["text"].str.strip() != ""]

# Zapisz połączone dane do pliku hate_speech_dataset.csv
combined_df.to_csv("hate_speech_dataset.csv", index=False)

print(f"Dane z BAN-PL.csv i BAN-PL2.csv zostały zapisane do hate_speech_dataset.csv")
print(f"Łączna liczba rekordów: {len(combined_df)}")

# Tokenizacja
def preprocess(text):
    # Konwertuj wartości nan/None/float na pusty string
    if pd.isna(text) or not isinstance(text, str):
        return []  # Zwracamy pustą listę dla wartości niebędących stringami
    
    text = text.lower()  # Zamiana na małe litery
    text = re.sub(r"[^a-ząćęłńóśźż ]", "", text)  # Usunięcie znaków specjalnych
    return text.split()

# Tworzenie słownika słów
word2index = {"<PAD>": 0, "<UNK>": 1}
index = 2

for text in df["text"]:
    words = preprocess(text)
    for word in words:
        if word not in word2index:
            word2index[word] = index
            index += 1

# Zamiana tekstu na liczby
def encode_text(text, max_len=10):
    words = preprocess(text)
    encoded = [word2index.get(word, 1) for word in words]  # 1 = <UNK>
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))  # Padding
    return encoded[:max_len]

df["encoded"] = df["text"].apply(lambda x: encode_text(x))

# Dataset dla PyTorch
class HateSpeechDataset(Dataset):
    def __init__(self, df):
        self.texts = torch.tensor(df["encoded"].tolist(), dtype=torch.long)
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

dataset = HateSpeechDataset(df)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

import torch.nn as nn
import torch.optim as optim

class HateSpeechModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=16):
        super(HateSpeechModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Ostatni krok LSTM
        return self.sigmoid(x)

# Inicjalizacja modelu
vocab_size = len(word2index)
model = HateSpeechModel(vocab_size)

# Ustawienia treningu
criterion = nn.BCELoss()  # Binary Cross Entropy dla klasyfikacji binarnej
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 12

for epoch in range(epochs): 
    total_loss = 0
    for texts, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(texts).squeeze(dim=1)  # Upewniamy się, że kształt pasuje
        loss = criterion(outputs, labels)  # Teraz kształty są zgodne
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoka {epoch+1}, Strata: {total_loss:.4f}")

# Zapisanie modelu
torch.save(model.state_dict(), "hate_speech_model.pth")

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Załaduj model
vocab_size = len(word2index)
model = HateSpeechModel(vocab_size)
model.load_state_dict(torch.load("hate_speech_model.pth", map_location=torch.device("cpu")))
model.eval()

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    
    # Tokenizacja i konwersja na tensor
    encoded = encode_text(text)
    tensor_input = torch.tensor([encoded], dtype=torch.long)

    # Predykcja
    with torch.no_grad():
        prediction = model(tensor_input).item()

    label = "hate" if prediction > 0.5 else "neutral"
    return jsonify({"text": text, "label": label, "score": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

