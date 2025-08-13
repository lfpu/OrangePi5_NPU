# Python FastAPI Web API with Docker

## RKNN running in orange pi 5

This project is a simple web API service using FastAPI, containerized with Docker.
and this project is for orange pi 5 RKNN model calculation
/models if the two models, one is for find faces, one is for get face features
## How to Build and Run with Docker

1. **Build the Docker image:**
   ```powershell
   docker build -t python-web-api .
   ```
2. **Run the Docker container:**
   ```powershell
   docker run -d -p 8000:8000 python-web-api
   ```
3. **Access the API:**
   Open your browser and go to [http://localhost:8000](http://localhost:8000)

## Development (without Docker)

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the app:
   ```powershell
   uvicorn main:app --reload
   ```

## API Endpoints
- `GET /` : Returns a hello world message.
