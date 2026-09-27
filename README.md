# Guitar Audio ML Microservice (FastAPI + Docker)

This repository contains a production-ready, containerized microservice for guitar audio preprocessing and feature extraction, designed for real-time Music Tech and Automated Chord Recognition (ACR) applications.

## Key Features
* **Advanced Audio Preprocessing:** Converts raw guitar audio (`.wav`, `.mp3`) into high-contrast 2D Log-Mel Spectrograms using `librosa` and `numpy`.
* **Slaney-Standard Bank Filters:** Implements area-normalized triangular filterbank processing, precisely modeling human and musical psychoacoustics.
* **Production-Ready FastAPI:** High-performance web-server with automated Swagger UI documentation and asynchronous file streaming.
* **Full Docker Isolation:** Cross-platform deployment via a lightweight `python:3.10-slim` Linux container with pre-configured native audio codecs (`libsndfile1`, `ffmpeg`).

## Technical Stack
* **Languages & Frameworks:** Python, FastAPI, PyTorch-ready Tensor Outputs, NumPy, Librosa.
* **Infrastructure:** Docker, WSL 2, Linux environment emulation.

## Production Tensor Output Format
The service dynamically resamples any incoming audio to **22050 Hz** and outputs a 4D tensor formatted for modern CV/Audio architectures (like CNNs, U-Net, Audio Spectrogram Transformers):
`[Batch_Size (1), Channels (1), Height/Mel_Bins (128), Width/Time_Frames (X)]`

## How to Run Locally

### 1. Using Docker (Recommended)
Build and run the Linux-isolated container with a single command:
```bash
docker build -t guitar-chord-service .
docker run -d -p 8000:8000 --name guitar_ai_container guitar-chord-service
```

### 2. Interactive Testing
Once started, open your browser and navigate to:
**http://127.0.0.1:8000**
Use the built-in Swagger UI to upload your guitar tracks and verify the output tensor shape.
