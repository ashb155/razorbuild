import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.inventory_service import InventoryService
from services.session_service import SessionManager
from services.razorpay_service import RazorpayService
from ai.intent_pipeline import SarvamIntentPipeline
from agent.dialog_manager import DialogManager
from models.schemas import ExtractedIntent

class MockIntentPipeline:
    """Mock intent pipeline for deterministic test scenarios without requiring GPU/GGUF model load."""
    def __init__(self, predefined_intents=None):
        self.intents = predefined_intents or {}

    def extract_intent(self, text: str, context: str = None):
        return self.intents.get(text, {"action": "unknown", "item": None, "requires_confirmation": False})

class TestRazorpayVoiceAgentSuite(unittest.TestCase):
    def setUp(self):
        self.inventory_service = InventoryService()
        self.session_manager = SessionManager()
        self.razorpay_service = RazorpayService()
        self.mock_ai = MockIntentPipeline()
        self.dialog_manager = DialogManager(
            inventory_service=self.inventory_service,
            session_manager=self.session_manager,
            razorpay_service=self.razorpay_service,
            intent_pipeline=self.mock_ai
        )
        self.phone = "+919999999999"

    def test_price_check_and_followup_add(self):
        """Test multi-turn dialog: User asks price of smartwatch, then confirms with 'हाँ वो अड कर दो'."""
        # Turn 1: Search
        self.mock_ai.intents["नॉइज कलर फिट प्रो कितने का है"] = {
            "action": "search",
            "item": "noise colorfit pro 4",
            "requires_confirmation": False
        }
        res1 = self.dialog_manager.process_utterance("नॉइज कलर फिट प्रो कितने का है", phone_number=self.phone)
        self.assertEqual(res1["status"], "success")
        self.assertIn("Noise ColorFit Pro 4", res1["message"])
        self.assertIn("2499", res1["message"])
        
        # Verify pending search state was set
        session = self.session_manager.get_session(self.phone)
        self.assertEqual(session.pending_search, "noise_smartwatch")

        # Turn 2: Coreference follow-up "हाँ वो अड कर दो"
        self.mock_ai.intents["हाँ वो अड कर दो"] = {
            "action": "add",
            "item": "that",
            "requires_confirmation": False
        }
        res2 = self.dialog_manager.process_utterance("हाँ वो अड कर दो", phone_number=self.phone)
        self.assertEqual(res2["status"], "success")
        self.assertIn("Noise ColorFit Pro 4", res2["message"])
        self.assertEqual(session.cart.get("noise_smartwatch"), 1)

    def test_decline_cross_sell(self):
        """Test user declining a cross-sell suggestion (e.g. 'सोनी नहीं चाहिए')."""
        session = self.session_manager.get_session(self.phone)
        session.cart["noise_smartwatch"] = 1
        session.pending_cross_sell = "sony_headphones"

        # User says "सोनी नहीं चाहिए" / "no thanks"
        self.mock_ai.intents["सोनी नहीं चाहिए"] = {
            "action": "remove",
            "item": "sony",
            "requires_confirmation": False
        }
        res = self.dialog_manager.process_utterance("सोनी नहीं चाहिए", phone_number=self.phone)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["action"], "decline_suggestion")
        self.assertIn("No problem", res["message"])
        # Ensure existing cart item was NOT deleted
        self.assertEqual(session.cart.get("noise_smartwatch"), 1)
        self.assertIsNone(session.pending_cross_sell)

    def test_out_of_stock_alternative_suggestion(self):
        """Test out of stock handling and alternative suggestion."""
        self.mock_ai.intents["add boat airdopes to cart"] = {
            "action": "add",
            "item": "boat earbuds",
            "quantity": 1,
            "requires_confirmation": False
        }
        res = self.dialog_manager.process_utterance("add boat airdopes to cart", phone_number=self.phone)
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["reason"], "out_of_stock")
        self.assertIn("out of stock", res["message"])
        self.assertIsNotNone(res.get("alternative"))

    def test_gated_checkout_and_otp_confirmation(self):
        """Test cart checkout triggering OTP confirmation gate and payment link replay."""
        session = self.session_manager.get_session(self.phone)
        session.cart["noise_smartwatch"] = 1

        # Step 1: Checkout request (should trigger OTP gate)
        self.mock_ai.intents["I want to checkout my cart"] = {
            "action": "checkout",
            "item": "cart",
            "requires_confirmation": True
        }
        res1 = self.dialog_manager.process_utterance("I want to checkout my cart", phone_number=self.phone)
        self.assertEqual(res1["status"], "pending_confirmation")
        self.assertIn("Confirm to generate your secure Razorpay payment link", res1["message"])
        self.assertIsNotNone(session.pending_confirmation)

        # Step 2: Confirmation passed (skip_gate=True)
        res2 = self.dialog_manager.process_utterance("Yes confirm", phone_number=self.phone, skip_gate=True)
        self.assertEqual(res2["status"], "success")
        self.assertEqual(res2["action"], "checkout_cart")
        self.assertIn("payment link", res2["message"].lower())
        self.assertEqual(len(session.cart), 0)  # Cart should be cleared after checkout link generation

    def test_merchant_invoice_generation(self):
        """Test B2B GST Invoice generation."""
        self.mock_ai.intents["send an invoice of 5000 to Acme Corp"] = {
            "action": "create_invoice",
            "amount": 5000.0,
            "company": "Acme Corp",
            "requires_confirmation": False
        }
        res = self.dialog_manager.process_utterance("send an invoice of 5000 to Acme Corp", phone_number=self.phone)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["action"], "create_invoice")
        self.assertIn("Acme Corp", res["message"])
        self.assertIn("5000", res["message"])

    def test_thirdwatch_fraud_rate_limiter(self):
        """Test Razorpay Thirdwatch fraud rate limiting upon repeated financial requests."""
        session = self.session_manager.get_session("fraud_test_user")
        self.mock_ai.intents["pay"] = {"action": "pay", "order_id": "order_123", "requires_confirmation": True}
        
        # Trigger 15 gated actions (reaching limit of 15)
        for _ in range(15):
            self.dialog_manager.process_utterance("pay", phone_number="fraud_test_user")
            
        # 16th action should trigger fraud alert
        res = self.dialog_manager.process_utterance("pay", phone_number="fraud_test_user")
        self.assertEqual(res["status"], "blocked")
        self.assertEqual(res["reason"], "rate_limit_exceeded")
        self.assertIn("Thirdwatch", res["message"])

if __name__ == "__main__":
    unittest.main()
