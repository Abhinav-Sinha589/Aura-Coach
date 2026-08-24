# -------- STAGE 1: FRONTEND BUILD (React) --------
FROM node:18-bullseye-slim AS frontend-build

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --legacy-peer-deps || npm install --legacy-peer-deps

# Build the React frontend
COPY public ./public
COPY src ./src
RUN npm run build


# -------- STAGE 2: BACKEND & SERVING (FastAPI) --------
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=10000

# Install Python requirements
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code, ML models, and metadata
COPY backend /app/backend

# Copy built frontend assets to /app/build (FastAPI will serve them automatically)
COPY --from=frontend-build /app/build /app/build

# Create data directory for session storage at runtime
RUN mkdir -p /app/data/raw

# Expose Render default port
EXPOSE 10000

# Run Uvicorn with dynamic port binding for Render
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
