import os
import onnxruntime as ort
import numpy as np

# Assuming the models are inside 'asrassets' in the backend directory
# e.g., asrassets/hi, asrassets/ta, etc.
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "asrassets"))

class IndicASRPipeline:
    def __init__(self):
        self.sessions = {}
        self.tokens = {}
        self.supported_languages = ["hi", "kn", "ml", "mr", "ta", "te"]
        
        print("Initializing Indic ASR Pipeline...")
        self._load_models()

    def _load_models(self):
        for lang in self.supported_languages:
            lang_dir = os.path.join(MODELS_DIR, lang)
            model_path = os.path.join(lang_dir, "model.int8.onnx")
            tokens_path = os.path.join(lang_dir, "tokens.txt")
            
            if os.path.exists(model_path) and os.path.exists(tokens_path):
                print(f"Loading {lang} model from {model_path}...")
                
                # Load the ONNX model
                # Using CPUExecutionProvider for standard CPU inference
                session_opts = ort.SessionOptions()
                session = ort.InferenceSession(model_path, sess_options=session_opts, providers=["CPUExecutionProvider"])
                self.sessions[lang] = session
                
                # Load tokens
                with open(tokens_path, "r", encoding="utf-8") as f:
                    # Depending on exactly how your tokens.txt is formatted, 
                    # you might need to adjust this parsing logic.
                    self.tokens[lang] = [line.strip() for line in f.readlines()]
                
                print(f"Successfully loaded {lang} (Tokens: {len(self.tokens[lang])})")
            else:
                print(f"Skipping {lang}: Model or tokens not found in {lang_dir}")

    def transcribe(self, lang: str, audio_array: np.ndarray) -> str:
        """
        Transcribes a 16kHz audio array using the loaded ONNX model.
        audio_array should be a 1D numpy array of floats.
        """
        if lang not in self.sessions:
            raise ValueError(f"Language '{lang}' is not loaded or supported.")
            
        session = self.sessions[lang]
        
        # NOTE: The exact preprocessing (like extracting Mel filterbanks) 
        # depends entirely on how the AI4Bharat Conformer was exported to ONNX.
        # Often, it expects acoustic features (e.g., shape [batch, time, 80]) 
        # or raw waveform (e.g., shape [batch, time]).
        
        # This is a placeholder for the actual inference call. You will need to 
        # format the input_feed to match what the ONNX model expects.
        print(f"Processing {len(audio_array)} samples for {lang}...")
        
        # input_name = session.get_inputs()[0].name
        # outputs = session.run(None, {input_name: audio_array_processed})
        # decoded_text = self._decode_output(outputs, self.tokens[lang])
        
        # return decoded_text
        return f"[MOCK TRANSCRIPT FOR {lang.upper()}] - Mujhe shoes chahiye"

    def _decode_output(self, outputs, vocab):
        # Implementation of CTC greedy decoding or similar based on model output
        pass

# Singleton instance for ASR
asr_pipeline = IndicASRPipeline()

class SarvamIntentPipeline:
    def __init__(self):
        print("Initializing Sarvam-1 Intent Pipeline...")
        self.model_path = os.path.join(os.path.dirname(__file__), "sarvam-1.Q4_K_M.gguf")
        self.llm = None
        self._load_model()
        
    def _load_model(self):
        try:
            from llama_cpp import Llama
            if os.path.exists(self.model_path):
                print(f"Loading Sarvam-1 GGUF from {self.model_path}...")
                self.llm = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)
                print("Successfully loaded Sarvam-1.")
            else:
                print(f"Skipping Sarvam-1: Model not found at {self.model_path}")
        except ImportError:
            print("Skipping Sarvam-1: llama-cpp-python is not installed.")
            
    def extract_intent(self, text: str) -> dict:
        if not self.llm:
            return {"error": "Sarvam-1 model not loaded."}
        
        
        # Prompt with examples to guide the LLM
        prompt = f"""Extract the user intent as a JSON object. Include the following fields when applicable:
- action (search, add, checkout, pay, remove, track, cancel, refund, create_payment_link, cancel_subscription, check_settlement, faq, create_offer, check_emi, create_invoice, payout, split_payment, create_qr, save_card, track_refund)
- item (the product or entity)
- quantity (default 1 if not mentioned)
- size (null if not mentioned)
- order_id (for payment, tracking, cancel, refund)
- amount (float, for payment links, refunds, invoices, payouts, or qr codes)
- sub_id (for subscriptions)
- topic (for faqs)
- discount_percent (integer, for offers)
- company (string, for invoices)
- recipient (string, for payouts or split payments)
- split_percent (integer, for split payments)
- card_network (string, for saving cards)
- cross_sell (string, suggest a complimentary item to upsell the user if they are adding/searching an item)
- requires_confirmation (boolean, true ONLY if the action involves spending/moving money like pay, cancel, refund, payout, split_payment, create_qr, create_payment_link. MUST be false for read-only actions like track, track_refund, check_settlement, search, faq, remove)

Examples:
User: "what is the status of order 123"
Result: {{ "action": "track", "order_id": "123", "requires_confirmation": false }}
User: "did my money settle yet?"
Result: {{ "action": "check_settlement", "requires_confirmation": false }}
User: "remove the black jacket from my cart"
Result: {{ "action": "remove", "item": "black jacket", "requires_confirmation": false }}
User: "create a 15 percent discount for Diwali"
Result: {{ "action": "create_offer", "discount_percent": 15, "requires_confirmation": true }}

User: "can I pay for these shoes in EMIs?"
Result: {{ "action": "check_emi", "item": "shoes" }}

User: "send an invoice of 5000 to Acme Corp"
Result: {{ "action": "create_invoice", "amount": 5000.0, "company": "Acme Corp", "requires_confirmation": true }}

User: "I want to checkout my cart"
Result: {{ "action": "checkout", "item": "cart", "requires_confirmation": true }}

User: "what is the price of white sneakers"
Result: {{ "action": "search", "item": "white sneakers", "cross_sell": "white socks" }}

User: "I need to pay for the order 12345"
Result: {{ "action": "pay", "order_id": "12345", "requires_confirmation": true }}

User: "refund my order 12345"
Result: {{ "action": "refund", "order_id": "12345", "requires_confirmation": true }}

User: "create a payment link for 1500.50"
Result: {{ "action": "create_payment_link", "amount": 1500.50, "requires_confirmation": true }}

User: "cancel subscription sub_999"
Result: {{ "action": "cancel_subscription", "sub_id": "sub_999", "requires_confirmation": true }}

User: "what are the charges for UPI?"
Result: {{ "action": "faq", "topic": "upi charges" }}

User: "मुझे मेरा ऑर्डर 98765 रिफंड चाहिए"
Result: {{ "action": "refund", "order_id": "98765", "requires_confirmation": true }}

User: "माझे सबस्क्रिप्शन sub_444 रद्द करा"
Result: {{ "action": "cancel_subscription", "sub_id": "sub_444", "requires_confirmation": true }}

User: "Pay my employee John Doe 50000 rupees"
Result: {{ "action": "payout", "amount": 50000.0, "recipient": "John Doe", "requires_confirmation": true }}

User: "Split order 123 and send 20 percent to vendor Y"
Result: {{ "action": "split_payment", "order_id": "123", "split_percent": 20, "recipient": "vendor Y", "requires_confirmation": true }}

User: "Create a BharatQR code for 500 rupees"
Result: {{ "action": "create_qr", "amount": 500.0, "requires_confirmation": true }}

User: "Save my HDFC credit card for future purchases"
Result: {{ "action": "save_card", "card_network": "HDFC" }}

User: "Track the refund status of order 456"
Result: {{ "action": "track_refund", "order_id": "456" }}

User: "{text}"
Result:"""
        print(f"Extracting intent for text: '{text}'...")
        response = self.llm(
            prompt,
            max_tokens=150,
            stop=["\n", "}"],
            echo=False,
            temperature=0.1
        )
        
        raw = response['choices'][0]['text']
        start = raw.find('{')
        end = raw.rfind('}')
        output_text = raw[start:end+1] if start != -1 and end != -1 else raw.strip()
        if not output_text.endswith("}"):
            output_text += "}"
            
        try:
            import json
            intent = json.loads(output_text)
            # Apply defaults
            intent.setdefault('quantity', 1)
            intent.setdefault('size', None)
            intent.setdefault('cross_sell', None)
            intent.setdefault('requires_confirmation', False)
            
            # Enforce gating on money actions
            if intent.get('action') in ['pay', 'refund', 'create_payment_link', 'cancel_subscription', 'create_offer', 'create_invoice']:
                intent['requires_confirmation'] = True
                
            # For pay action, keep only order_id (remove item if present)
            if intent.get('action') == 'pay' and 'order_id' in intent:
                intent.pop('item', None)
            return intent
        except Exception as e:
            import ast, re
            try:
                intent = ast.literal_eval(output_text)
                if isinstance(intent, dict):
                    intent.setdefault('quantity', 1)
                    intent.setdefault('size', None)
                    intent.setdefault('cross_sell', None)
                    intent.setdefault('requires_confirmation', False)
                    
                    if intent.get('action') in ['pay', 'refund', 'create_payment_link', 'cancel_subscription', 'create_offer', 'create_invoice']:
                        intent['requires_confirmation'] = True
                        
                    if intent.get('action') == 'pay' and 'order_id' in intent:
                        intent.pop('item', None)
                    return intent
            except Exception:
                pass
            
            print(f"Failed to parse JSON from Sarvam: {output_text}")
            
            # Fallback regex parsing for robustness
            lowered = text.lower()
            intent = {"action": "unknown", "item": "unknown"}

            # refund intent
            refund_match = re.search(r'refund\s+.*?order\s+([a-zA-Z0-9_]+)', lowered)
            if refund_match:
                intent["action"] = "refund"
                intent["order_id"] = refund_match.group(1)
                intent["requires_confirmation"] = True
                intent.pop("item", None)
                return intent
                
            # payment link intent
            link_match = re.search(r'payment\s+link\s+for\s+([\d\.]+)', lowered)
            if link_match:
                intent["action"] = "create_payment_link"
                intent["amount"] = float(link_match.group(1))
                intent["requires_confirmation"] = True
                intent.pop("item", None)
                return intent
                
            # subscription intent
            sub_match = re.search(r'cancel\s+subscription\s+([a-zA-Z0-9_]+)', lowered)
            if sub_match:
                intent["action"] = "cancel_subscription"
                intent["sub_id"] = sub_match.group(1)
                intent["requires_confirmation"] = True
                intent.pop("item", None)
                return intent

            # faq intent
            faq_match = re.search(r'charges\s+for\s+(.+)', lowered)
            if faq_match:
                intent["action"] = "faq"
                intent["topic"] = faq_match.group(1).strip("?")
                intent.pop("item", None)
                return intent

            # offer intent
            offer_match = re.search(r'create\s+a\s+(\d+)\s*(?:%|percent)\s+discount', lowered)
            if offer_match:
                intent["action"] = "create_offer"
                intent["discount_percent"] = int(offer_match.group(1))
                intent["requires_confirmation"] = True
                intent.pop("item", None)
                return intent

            # invoice intent
            invoice_match = re.search(r'invoice\s+of\s+([\d\.]+)\s+to\s+(.+)', lowered)
            if invoice_match:
                intent["action"] = "create_invoice"
                intent["amount"] = float(invoice_match.group(1))
                intent["company"] = invoice_match.group(2).strip()
                intent["requires_confirmation"] = True
                intent.pop("item", None)
                return intent
                
            # emi intent
            if "emi" in lowered:
                intent["action"] = "check_emi"
                intent.pop("item", None)
                return intent

            # standard payment intent
            pay_match = re.search(r'pay\s+(?:for\s+)?order\s+(\d+)', lowered)
            if pay_match:
                intent["action"] = "pay"
                intent["order_id"] = pay_match.group(1)
                intent["requires_confirmation"] = True
                intent.pop("item", None)
                return intent

            # generic search fallback
            search_match = re.search(r'([^\s]+)\s+(.+)', lowered)
            if search_match:
                intent["action"] = "search"
                intent["item"] = search_match.group(2).strip()
                intent["quantity"] = 1
                intent["size"] = None
                return intent

            # If still failing, return unknown with raw output
            return {"action": "unknown", "item": "unknown", "raw_output": output_text}

intent_pipeline = SarvamIntentPipeline()

if __name__ == "__main__":
    pass
