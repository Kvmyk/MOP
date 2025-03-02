# Monitorowanie Obraźliwych Przekazów (MOP) 🚨🛡️

**Monitorowanie Obraźliwych Przekazów (MOP)** to projekt mający na celu automatyczne wykrywanie i klasyfikację obraźliwych treści w języku polskim. Wykorzystuje on model oparty na sieciach neuronowych do analizy tekstu i identyfikacji potencjalnie szkodliwych komunikatów.

## ✨ Funkcje projektu

- **Czyszczenie tekstu**: Usuwanie niepotrzebnych znaków i formatowań, takich jak wzmianki użytkowników, nazwiska czy URL-e.
- **Przetwarzanie wstępne**: Tokenizacja, normalizacja oraz tworzenie słownika słów na podstawie dostarczonych danych.
- **Kodowanie tekstu**: Konwersja przetworzonych słów na ich reprezentacje numeryczne z użyciem wcześniej utworzonego słownika.
- **Model klasyfikacji**: Implementacja modelu LSTM do klasyfikacji tekstu jako "hate" lub "neutral".
- **API**: Udostępnienie modelu poprzez interfejs API zbudowany na Flasku, umożliwiający analizę nowych tekstów.

## 📂 Struktura projektu

- `model.py` – Główny plik zawierający implementację modelu oraz funkcje przetwarzania danych.
- `hate_speech_dataset.csv` – Zbiór danych używany do trenowania modelu.
- `hate_speech_model.pth` – Zapis wytrenowanego modelu.
- `hate_speech_model_best.pth` – Najlepsza wersja modelu uzyskana podczas treningu.

## 📊 Wykorzystany zbiór danych

Projekt korzysta z [BAN-PL: Polish Dataset of Banned Harmful and Offensive Content from Wykop.pl Web Service](https://github.com/ZILiAT-NASK/BAN-PL), który jest pierwszym publicznie dostępnym zbiorem danych zawierającym treści uznane za szkodliwe i obraźliwe przez profesjonalnych moderatorów serwisu Wykop.pl.  
Zbiór ten składa się z 48 000 próbek zanonimizowanych treści, z czego 24 000 to treści szkodliwe, a 24 000 to treści neutralne.

## 🚀 Jak uruchomić projekt

1. **Klonuj repozytorium**:
   ```bash
   git clone https://github.com/Kvmyk/MOP.git
   ```
2. **Zainstaluj wymagane biblioteki**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Uruchom API**:
   ```bash
   python model.py
   ```
   API będzie dostępne pod adresem `http://0.0.0.0:5001`.

## 🌐 Korzystanie z API

Aby skorzystać z API, wyślij żądanie **POST** na endpoint `/analyze` z treścią w formacie JSON:

```json
{
  "text": "Twój tekst do analizy"
}
```

**Odpowiedź zawiera**:
- `text` – Analizowany tekst.
- `label` – Etykieta klasyfikacji (`hate` lub `neutral`).
- `score` – Wynik predykcji modelu.
- `confidence` – Pewność predykcji na skali od 0 do 1.

## 📝 Przykład użycia

```bash
curl -X POST http://0.0.0.0:5001/analyze -H "Content-Type: application/json" -d '{"text": "Przykładowy tekst do analizy"}'
```

Odpowiedź:
```json
{
  "text": "Przykładowy tekst do analizy",
  "label": "neutral",
  "score": 0.234,
  "confidence": 0.532
}
```

## ✍️ Uwagi końcowe

MOP to narzędzie stworzone z myślą o wspomaganiu procesu moderacji treści w języku polskim. Dzięki wykorzystaniu zaawansowanych technik przetwarzania języka naturalnego oraz dostępu do rzeczywistych danych, projekt stanowi solidną podstawę do dalszego rozwoju i integracji w systemach monitorowania treści online.

---

> **Uwaga**: Projekt korzysta z danych zanonimizowanych i jest przeznaczony wyłącznie do celów badawczych.
