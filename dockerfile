# -------- FRONTEND BUILD (React) --------
FROM node:18 AS frontend-build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build


# -------- BACKEND (FastAPI) --------
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY --from=frontend-build /app/build /app/backend/build

# Models are already trained and committed in backend/models/*.joblib,
# so no build-time training step is needed.

EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
