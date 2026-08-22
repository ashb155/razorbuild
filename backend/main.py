from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import soundfile as sf
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_pipeline import asr_pipeline
from checkout_agent import process_voice_command

app = FastAPI(title="Razorpay Voice SDK Backend")

@app.get("/")
def read_root():
    return {"status": "Indic Voice Backend is running", "supported_languages": asr_pipeline.supported_languages}

@app.post("/api/process-voice")
async def process_voice(
    audio: UploadFile = File(...),
    language: str = Form(...)
):
    """
    Endpoint that the Android SDK will call.
    Receives an audio file (e.g., .wav or .ogg) and the user's language code.
    """
    if language not in asr_pipeline.supported_languages:
        return JSONResponse(status_code=400, content={"error": f"Language '{language}' not supported. Supported: {asr_pipeline.supported_languages}"})
    
    try:
        # 1. Read audio bytes
        audio_bytes = await audio.read()
        
        # 2. Decode audio into numpy array
        # Assuming the Android SDK sends a format soundfile can read (like WAV or OGG)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Ensure it's mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
            
        # NOTE: If sample_rate is not 16000 (standard for ASR), you would 
        # need to resample it here using librosa.resample.
        
        # 3. Transcribe using the loaded ONNX model
        transcribed_text = asr_pipeline.transcribe(language, audio_data)
        
        # 4. Pass the transcribed text into our Razorpay Agent Execution Engine
        agent_response = process_voice_command(transcribed_text)
        
        # 5. Return the unified JSON response to the Android SDK
        return {
            "status": "success",
            "transcription": transcribed_text,
            "agent_response": agent_response
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
