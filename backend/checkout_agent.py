"""
Razorpay Voice AI Agent - Checkout & Operations Engine
======================================================
Unified conversational execution engine for customer shopping,
cart management, OTP confirmation gating, refunds, and merchant operations.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RazorpayAgent")

# Import modular services & agent
from services.inventory_service import InventoryService
from services.session_service import session_manager
from services.razorpay_service import razorpay_service
from ai.intent_pipeline import intent_pipeline
from agent.dialog_manager import DialogManager

# Initialize shared singleton instances
inventory_service = InventoryService()
dialog_manager = DialogManager(
    inventory_service=inventory_service,
    session_manager=session_manager,
    razorpay_service=razorpay_service,
    intent_pipeline=intent_pipeline
)

# Backward-compatibility accessor for inventory
def load_inventory():
    return inventory_service.get_raw_inventory()

# Backward-compatibility accessor for MOCK_CART dictionary
class MockCartAdapter(dict):
    """Adapter to let legacy callers access cart via MOCK_CART[phone_number]."""
    def __getitem__(self, key):
        return session_manager.get_session(key).cart

    def get(self, key, default=None):
        session = session_manager.get_session(key)
        return session.cart if session else default

MOCK_CART = MockCartAdapter()

def process_voice_command(
    query: str,
    phone_number: Optional[str] = None,
    skip_gate: bool = False,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Processes a voice transcription or text query through the Razorpay Voice Agent pipeline.
    
    Args:
        query: Transcribed user speech or text input.
        phone_number: Customer identifier / session key.
        skip_gate: Flag indicating user just confirmed the OTP / gated money action.
        context: Prior conversation history for context-aware intent extraction.
        
    Returns:
        Structured response dictionary containing status, action, message, and payment links.
    """
    logger.info(f"Processing query: '{query}' (session: {phone_number}, skip_gate: {skip_gate})")
    response = dialog_manager.process_utterance(
        query=query,
        phone_number=phone_number,
        skip_gate=skip_gate,
        context=context
    )
    return response

if __name__ == "__main__":
    sentences = [
        # English
        "what is the price of white sneakers",
        "add red shoes to my cart",
        "remove the black jacket from my cart",
        "track order_TTKQEPzDQsai1H",
        "cancel order order_TTKQEPzDQsai1H",
        "can I pay for these shoes in EMIs?",
        "I need to pay for order order_TTKQEPzDQsai1H",
        "refund my order order_TTKQEPzDQsai1H",
        "create a payment link for 2000 rupees",
        "cancel subscription sub_999",
        "did my money settle yet?",
        "create a 15 percent discount for Diwali",
        "send an invoice of 5000 to Acme Corp",
        "Pay my employee John Doe 50000 rupees",
        "Split order order_TTKQEPzDQsai1H and send 20 percent to vendor Y",
        "Create a BharatQR code for 500 rupees",
        "Save my HDFC credit card for future purchases",
        "Track the refund status of order order_TTKQEPzDQsai1H",
        "what are the charges for UPI?",
        "refund 500 rupees from order order_TTKQEPzDQsai1H",
        "my payment failed",
        "get my magic checkout address",
        "what are the current offers",

        # Hindi (hi)
        "सफेद स्नीकर्स की कीमत क्या है?",
        "लाल जूते मेरे कार्ट में जोड़ें",
        "मेरे कार्ट से काली जैकेट हटा दें",
        "ऑर्डर order_TTKQEPzDQsai1H को ट्रैक करें",
        "ऑर्डर order_TTKQEPzDQsai1H रद्द करें",
        "क्या मैं इन जूतों के लिए ईएमआई में भुगतान कर सकता हूँ?",
        "मुझे ऑर्डर order_TTKQEPzDQsai1H के लिए भुगतान करना है",
        "मेरा ऑर्डर order_TTKQEPzDQsai1H रिफंड करें",
        "2000 रुपये के लिए पेमेंट लिंक बनाएं",
        "सब्सक्रिप्शन sub_999 रद्द करें",
        "क्या मेरा पैसा सेटल हो गया?",
        "दिवाली के लिए 15 प्रतिशत डिस्काउंट बनाएं",
        "Acme Corp को 5000 का इनवॉइस भेजें",
        "मेरे कर्मचारी John Doe को 50000 रुपये का भुगतान करें",
        "ऑर्डर order_TTKQEPzDQsai1H को स्प्लिट करें और 20 प्रतिशत vendor Y को भेजें",
        "500 रुपये के लिए भारत क्यूआर कोड बनाएं",
        "भविष्य की खरीदारी के लिए मेरा एचडीएफसी क्रेडिट कार्ड सेव करें",
        "ऑर्डर order_TTKQEPzDQsai1H के रिफंड स्टेटस को ट्रैक करें",
        "यूपीआई के लिए क्या चार्ज हैं?",
        "ऑर्डर order_TTKQEPzDQsai1H से 500 रुपये रिफंड करें",
        "मेरा भुगतान विफल हो गया",
        "मेरा मैजिक चेकआउट पता प्राप्त करें",
        "वर्तमान ऑफ़र क्या हैं"
    ]
    
    print(f"\n--- Running Multi-lingual Intent Test Suite ({len(sentences)} cases) ---\n")
    for s in sentences:
        res = process_voice_command(s, phone_number="+918888888888")
        print(f"User: '{s}' -> Status: {res.get('status')} | Message: {res.get('message')[:70]}...")
