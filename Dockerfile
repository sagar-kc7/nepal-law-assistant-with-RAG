# Start from a slim Python base image — smaller than the full python:3.12 image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first — this is a deliberate ordering trick for build caching.
# Docker caches each layer; if requirements.txt hasn't changed, this layer is reused
# on rebuilds instead of reinstalling everything, saving a lot of time during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY app/ ./app/
COPY data/nepal_legal_db_v3_contextual/ ./data/nepal_legal_db_v3_contextual/
COPY data/nepal_legal_raptor_tree/ ./data/nepal_legal_raptor_tree/

# Document which port the container listens on (informational — doesn't actually publish it)
EXPOSE 8000

# The command that runs when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]