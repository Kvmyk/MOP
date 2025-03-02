# Użyj oficjalnego obrazu Pythona jako bazowego
FROM python:3.9-slim

# Ustawienie katalogu roboczego
WORKDIR /app

# Skopiowanie plików aplikacji do katalogu roboczego
COPY . /app

# Instalacja zależności Pythona
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Eksponowanie portu, na którym działa aplikacja
EXPOSE 5001

# Komenda uruchamiająca aplikację
CMD ["python", "model.py"]
