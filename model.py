import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flask import Flask, request, jsonify

# Funkcja do czyszczenia tekstu
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

# Sprawdź czy trzeba wczytać dane - wykonaj tylko raz
do_data_loading = False  # Zmień na True gdy chcesz załadować dane

if do_data_loading:
    print("Wczytywanie plików...")

    # Wczytaj istniejące dane z hate_speech_dataset.csv
    try:
        hate_speech_df = pd.read_csv("hate_speech_dataset.csv")
        print(f"Wczytano {len(hate_speech_df)} rekordów z hate_speech_dataset.csv")
    except FileNotFoundError:
        # Jeśli plik nie istnieje, utwórz pusty DataFrame
        hate_speech_df = pd.DataFrame(columns=["text", "label"])
        print("Utworzono nowy DataFrame dla hate_speech_dataset.csv")

    # Wczytaj dane z BAN-PL2.csv
    ban_pl2_df = pd.read_csv("BAN-PL2.csv")
    print(f"Wczytano {len(ban_pl2_df)} rekordów z BAN-PL2.csv")

    # Wybierz tylko kolumny Text i Class i utwórz jawną kopię
    ban_pl2_data = ban_pl2_df[["Text", "Class"]].copy()

    # Zmień nazwy kolumn na odpowiadające hate_speech_dataset.csv
    ban_pl2_data.columns = ["text", "label"]

    # Zastosuj funkcję czyszczącą na kolumnie text
    print("Czyszczenie danych z BAN-PL2.csv...")
    ban_pl2_data["text"] = ban_pl2_data["text"].apply(clean_text)

    # Upewnij się, że wszystkie wartości w kolumnie tekst są stringami
    ban_pl2_data["text"] = ban_pl2_data["text"].astype(str)

    # Usuń puste wiersze (te, w których tekst jest pusty)
    ban_pl2_data = ban_pl2_data[ban_pl2_data["text"].str.strip() != ""]
    print(f"Po czyszczeniu pozostało {len(ban_pl2_data)} rekordów z BAN-PL2.csv")

    # Połącz dane
    combined_df = pd.concat([hate_speech_df, ban_pl2_data], ignore_index=True)
    print(f"Łączna liczba rekordów po połączeniu: {len(combined_df)}")

    # Zapisz połączone dane do pliku hate_speech_dataset.csv
    combined_df.to_csv("hate_speech_dataset.csv", index=False)
    print(f"Dane zostały zapisane do hate_speech_dataset.csv")
    print(f"Dodano {len(ban_pl2_data)} nowych rekordów z BAN-PL2.csv")

# Wczytanie danych dla treningu - wykonaj zawsze
df = pd.read_csv("hate_speech_dataset.csv")
print(f"Wczytano {len(df)} rekordów z hate_speech_dataset.csv do treningu.")

# Tokenizacja - zachowujemy więcej informacji
def preprocess(text):
    # Konwertuj wartości nan/None/float na pusty string
    if pd.isna(text) or not isinstance(text, str):
        return []  # Zwracamy pustą listę dla wartości niebędących stringami
    
    text = text.lower()  # Zamiana na małe litery
    # Zachowujemy więcej znaków dla lepszej precyzji - interpunkcja może być istotna
    text = re.sub(r"[^a-ząćęłńóśźż.,!?;: ]", "", text)
    return text.split()

# Tworzenie słownika słów bez limitów
word2index = {"<PAD>": 0, "<UNK>": 1}
index = 2

for text in df["text"]:
    words = preprocess(text)
    for word in words:
        if word not in word2index:
            word2index[word] = index
            index += 1

# Zamiana tekstu na liczby - zwiększamy maksymalną długość sekwencji
def encode_text(text, max_len=20):  # zwiększone z 10 do 20
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

# Zwiększamy batch_size dla lepszej stabilności treningu
dataset = HateSpeechDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model w mniejszej wersji
class HateSpeechModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=32):  # zmniejszono wymiary
        super(HateSpeechModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # usunięto bidirectional i zmniejszono złożoność
        self.fc = nn.Linear(hidden_dim, 1)  # usunięto dodatkową warstwę
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Ostatni krok LSTM
        x = self.fc(x)
        return self.sigmoid(x)

# Inicjalizacja modelu
vocab_size = len(word2index)
model = HateSpeechModel(vocab_size)

# Ustawienia treningu z mniejszym learning rate dla większej precyzji
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Dodano regularyzację L2
epochs = 30  # zwiększono liczbę epok

# Dodajemy early stopping
best_loss = float('inf')
patience = 5
counter = 0

# Zatrzymaj gdy accuracy jest wysoka i stabilna przez kilka epok
accuracy_history = []

for epoch in range(epochs):
    model.train()  # Explicitly set training mode
    total_loss = 0
    for texts, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(texts).squeeze(dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    # Validate after each epoch
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:  # Using training data as validation for now
            outputs = model(texts).squeeze(dim=1)
            val_loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(dataloader)
    accuracy = 100 * correct / total
    
    print(f"Epoka {epoch+1}, Strata: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.2f}%")
    
    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        # Save best model
        torch.save(model.state_dict(), "hate_speech_model_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Zatrzymaj gdy accuracy jest wysoka i stabilna przez kilka epok
    accuracy_history.append(accuracy)
    if len(accuracy_history) >= 3:  # Sprawdź ostatnie 3 epoki
        if all(acc > 98 for acc in accuracy_history[-3:]):  # Wszystkie ponad 98%
            if max(accuracy_history[-3:]) - min(accuracy_history[-3:]) < 0.5:  # Stabilne (różnica < 0.5%)
                print("Wysoka i stabilna accuracy osiągnięta. Zatrzymuję trening.")
                torch.save(model.state_dict(), "hate_speech_model_best.pth")
                break

# Ładujemy najlepszy model
model.load_state_dict(torch.load("hate_speech_model_best.pth"))

# Zapisanie modelu bez kwantyzacji dla zachowania maksymalnej precyzji
torch.save(model.state_dict(), "hate_speech_model.pth")

# API Flask z pełną precyzją
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    
    # Tokenizacja i konwersja na tensor
    encoded = encode_text(text)
    tensor_input = torch.tensor([encoded], dtype=torch.long)

    # Predykcja z pełną precyzją
    model.eval()  # Upewniamy się, że model jest w trybie ewaluacji
    with torch.no_grad():
        prediction = model(tensor_input).item()

    # Więcej szczegółowych informacji w odpowiedzi
    confidence = abs(prediction - 0.5) * 2  # Przekształcenie na skalę 0-1
    label = "hate" if prediction > 0.5 else "neutral"
    return jsonify({
        "text": text, 
        "label": label, 
        "score": prediction,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

