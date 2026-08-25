from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import soundfile as sf
import librosa
import io
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_pipeline import asr_pipeline
from checkout_agent import process_voice_command

app = FastAPI(title="Razorpay Voice SDK Backend")

# Allow Web SDK to call from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Indic Voice Backend is running", "supported_languages": asr_pipeline.supported_languages}

@app.get("/api/inventory")
def get_inventory():
    inventory_path = os.path.join(os.path.dirname(__file__), 'inventory.json')
    try:
        with open(inventory_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/cart")
async def get_cart():
    # Import MOCK_CART from checkout_agent
    from checkout_agent import MOCK_CART
    return MOCK_CART

@app.post("/api/process-voice")
async def process_voice(
    audio: UploadFile = File(None),
    text: str = Form(None),
    language: str = Form(...),
    phone_number: str = Form(None)
):
    """
    Endpoint that the Android SDK will call.
    Receives an audio file (e.g., .wav or .ogg) or text, and the user's language code.
    """
    if language not in asr_pipeline.supported_languages:
        return JSONResponse(status_code=400, content={"error": f"Language '{language}' not supported. Supported: {asr_pipeline.supported_languages}"})
    
    try:
        if text:
            transcription = text
            print(f"[DIRECT TEXT] {transcription}")
        elif audio:
            # 1. Read audio bytes
            audio_bytes = await audio.read()
            
            # 2. Decode audio into numpy array (Expects standard WAV from Web SDK)
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Ensure it's mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
                
            # Resample to 16kHz required by IndicConformer
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                
            # 3. Transcribe
            transcription = asr_pipeline.transcribe(language, audio_data)
        else:
            return JSONResponse(status_code=400, content={"error": "Must provide either audio or text."})
        
        # 4. Pass the transcribed text into our Razorpay Agent Execution Engine
        agent_response = process_voice_command(transcription, phone_number=phone_number)
        
        # 5. Return the unified JSON response to the Android SDK
        return {
            "status": "success",
            "transcription": transcription,
            "agent_response": agent_response
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/webhooks/razorpay")
async def razorpay_webhook(payload: dict):
    """
    Simulated Razorpay Webhook endpoint.
    Receives async events like payment.failed, payment.captured, refund.processed.
    """
    event = payload.get("event", "unknown_event")
    print(f"\n[WEBHOOK] Received async event from Razorpay: {event}")
    print(f"[WEBHOOK] Payload: {payload}")
    
    # In a real app, this is where we'd push an FCM notification to the Android device
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
