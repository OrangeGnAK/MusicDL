import io
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="Guitar Chord Classifier Service")

@app.post("/process-audio")
async def process_audio(file: UploadFile = File()):


    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        return JSONResponse(status_code=400, content={"error" : "Invalid file format"})

    try:
        audio_bytes = await file.read()

        audio_stream = io.BytesIO(audio_bytes)
        audio, sampling_rate = librosa.load(audio_stream, sr=22050)

        duration = len(audio) / sampling_rate

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sampling_rate, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        input_tensor = np.expand_dims(mel_spec_db, axis=(0, 1))

        return {
            "filename": file.filename,
            "duration_seconds": round(duration, 2),
            "sampling_rate": sampling_rate,
            "tensor_shape": list(input_tensor.shape)
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"File processing error: {str(e)}"})
