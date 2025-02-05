# Usa un'immagine base con Python
FROM python:3.12

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia il contenuto della directory corrente nel container
COPY . .

RUN pip install --upgrade pip

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Esegui lo script Python quando il container parte
CMD ["python", "docker-main.py"]
