import os
import io
import sys
import logging
from typing import Optional
import uvicorn
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.asr_pipeline import asr_pipeline
from services.inventory_service import InventoryService
from services.session_service import session_manager
from checkout_agent import process_voice_command

logger = logging.getLogger("RazorpayVoiceAPI")
inventory_service = InventoryService()

app = FastAPI(
    title="Razorpay Voice AI Assistant API",
    description="Multilingual Voice AI agent for Conversational Commerce & Razorpay Checkout integration.",
    version="2.0.0"
)

# Allow Web & Mobile SDK to call from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "Razorpay Indic Voice Commerce Backend is running",
        "version": "2.0.0",
        "supported_languages": asr_pipeline.supported_languages
    }

@app.get("/api/inventory")
def get_inventory():
    """Returns the live product catalog for the frontend storefront."""
    try:
        return inventory_service.get_raw_inventory()
    except Exception as e:
        logger.error(f"Failed to fetch inventory: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/cart")
def get_cart(phone_number: str = "+918888888888"):
    """Returns the active session's cart items as a flat {product_key: quantity} mapping."""
    session = session_manager.get_session(phone_number)
    return dict(session.cart)

@app.post("/api/process-voice")
async def process_voice(
    audio: UploadFile = File(None),
    text: str = Form(None),
    language: str = Form("en"),
    phone_number: str = Form(None),
    skip_gate: bool = Form(False),
    context: str = Form(None)
):
    """
    Core Voice & Text Processing Endpoint.
    Receives voice audio (WAV/OGG) or text, transcribes Indic speech, and executes intent.
    """
    if language not in asr_pipeline.supported_languages:
        return JSONResponse(
            status_code=400,
            content={"error": f"Language '{language}' not supported. Supported: {asr_pipeline.supported_languages}"}
        )
    
    try:
        if text:
            transcription = text
        elif audio:
            audio_bytes = await audio.read()
            if not audio_bytes:
                return JSONResponse(status_code=400, content={"error": "Empty audio data received."})
                
            try:
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Invalid audio format: {str(e)}"})
            
            if len(audio_data) == 0:
                return JSONResponse(status_code=400, content={"error": "Audio contains no frames."})
            
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
                
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                
            transcription = asr_pipeline.transcribe(language, audio_data)
        else:
            return JSONResponse(status_code=400, content={"error": "Must provide either audio or text."})
        
        # Execute conversational turn through Dialog Manager
        effective_context = "" if skip_gate else context
        agent_response = process_voice_command(
            query=transcription,
            phone_number=phone_number,
            skip_gate=skip_gate,
            context=effective_context
        )
        
        return {
            "status": "success",
            "transcription": transcription,
            "agent_response": agent_response
        }
        
    except Exception as e:
        logger.exception("Error processing voice request")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/webhooks/razorpay")
async def razorpay_webhook(payload: dict):
    """Simulated Razorpay Webhook endpoint for async event ingestion."""
    event = payload.get("event", "unknown_event")
    logger.info(f"Webhook received from Razorpay: {event}")
    return {"status": "ok", "event": event}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
