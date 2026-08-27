import os
import onnxruntime as ort
import numpy as np
import librosa

# Assuming the models are inside 'asrassets' in the backend directory
# e.g., asrassets/hi, asrassets/ta, etc.
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "asrassets"))

class IndicASRPipeline:
    def __init__(self):
        self.sessions = {}
        self.tokens = {}
        self.supported_languages = ["hi", "kn"]
        
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
                session_opts.intra_op_num_threads = 2
                session = ort.InferenceSession(model_path, sess_options=session_opts, providers=["CPUExecutionProvider"])
                self.sessions[lang] = session
                
                # Load tokens
                with open(tokens_path, "r", encoding="utf-8") as f:
                    # Token format is "<token> <id>"
                    self.tokens[lang] = [line.strip().split()[0] for line in f.readlines()]
                
                print(f"Successfully loaded {lang} (Tokens: {len(self.tokens[lang])})")
            else:
                print(f"Skipping {lang}: Model or tokens not found in {lang_dir}")

    def transcribe(self, lang: str, audio_array: np.ndarray) -> str:
        if lang not in self.sessions:
            raise ValueError(f"Language '{lang}' is not loaded or supported.")
            
        session = self.sessions[lang]
        
        # 1. Audio Preprocessing (Mel Spectrogram for NeMo)
        # Pre-emphasis
        if audio_array is None or len(audio_array) == 0:
            print(f"[ASR - {lang.upper()}] Empty audio array received, returning empty string.")
            return ""
            
        if len(audio_array) > 0:
            audio_array = np.append(audio_array[0], audio_array[1:] - 0.97 * audio_array[:-1])
            
        melspec = librosa.feature.melspectrogram(
            y=audio_array, 
            sr=16000, 
            n_fft=512, 
            hop_length=160, 
            win_length=320,
            window='hann',
            center=True,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0
        )
        # NeMo uses log(x + 1e-5)
        log_melspec = np.log(melspec + 1e-5).astype(np.float32)
        
        # Per-utterance mean/variance normalization
        mean = np.mean(log_melspec, axis=1, keepdims=True)
        std = np.std(log_melspec, axis=1, keepdims=True)
        log_melspec = (log_melspec - mean) / (std + 1e-5)
        
        audio_signal = np.expand_dims(log_melspec, axis=0) # Shape: [1, 80, time]
        length_val = np.array([audio_signal.shape[2]], dtype=np.int64) # Shape: [1]
        
        # 2. ONNX Inference
        inputs = {
            "audio_signal": audio_signal,
            "length": length_val
        }
        outputs = session.run(None, inputs)
        logprobs = outputs[0] # Shape: [1, time, vocab_size]
        
        # 3. CTC Decoding
        decoded_text = self._decode_output(logprobs, self.tokens[lang])
        
        print(f"[ASR - {lang.upper()}] Transcribed: '{decoded_text}'")
        return decoded_text

    def _decode_output(self, logprobs, vocab):
        predictions = np.argmax(logprobs[0], axis=-1)
        
        decoded_tokens = []
        prev_token = None
        # NeMo typically uses the last token as blank, or explicitly '<blk>'
        blank_id = len(vocab) - 1
        if "<blk>" in vocab:
            blank_id = vocab.index("<blk>")
            
        for token_id in predictions:
            if token_id != prev_token and token_id != blank_id:
                if token_id < len(vocab):
                    decoded_tokens.append(vocab[token_id])
            prev_token = token_id
            
        # Join and handle sentencepiece spaces
        text = "".join(decoded_tokens)
        text = text.replace("\u2581", " ").replace("<unk>", "").strip()
        return text

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
            from llama_cpp import Llama, LlamaGrammar
            import json
            if os.path.exists(self.model_path):
                print(f"Loading Sarvam-1 GGUF from {self.model_path}...")
                self.llm = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)
                
                schema = {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "item": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "size": {"type": "string"},
                        "order_id": {"type": "string"},
                        "amount": {"type": "number"},
                        "sub_id": {"type": "string"},
                        "topic": {"type": "string"},
                        "discount_percent": {"type": "integer"},
                        "company": {"type": "string"},
                        "recipient": {"type": "string"},
                        "split_percent": {"type": "integer"},
                        "card_network": {"type": "string"},
                        "cross_sell": {"type": "string"}
                    },
                    "required": ["action"]
                }
                self.grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
                
                print("Successfully loaded Sarvam-1.")
            else:
                print(f"Skipping Sarvam-1: Model not found at {self.model_path}")
        except ImportError:
            print("Skipping Sarvam-1: llama-cpp-python is not installed.")
            
    def _enforce_intent_rules(self, intent: dict) -> dict:
        intent.setdefault('quantity', 1)
        intent.setdefault('size', None)
        intent.setdefault('cross_sell', None)
        intent.setdefault('requires_confirmation', False)
        
        # Enforce gating on money actions
        if intent.get('action') in ['pay', 'refund', 'partial_refund', 'create_payment_link', 'cancel_subscription', 'create_offer', 'create_invoice', 'payout', 'split_payment', 'create_qr']:
            intent['requires_confirmation'] = True
            
        # For pay action, keep only order_id (remove item if present)
        if intent.get('action') == 'pay' and 'order_id' in intent:
            intent.pop('item', None)
        return intent
            
    def extract_intent(self, text: str, context: str = None) -> dict:
        if not self.llm:
            return {"error": "Sarvam-1 model not loaded."}
        
        
        # Prompt with examples to guide the LLM
        prompt = f"""Extract the user intent as a JSON object. Include the following fields when applicable:
- action (search, add, checkout, pay, remove, track, cancel, refund, create_payment_link, cancel_subscription, check_settlement, faq, create_offer, check_emi, create_invoice, payout, split_payment, create_qr, save_card, track_refund, handle_failed_payment, magic_checkout_address, check_offers, partial_refund, confirm_cart)
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
- requires_confirmation (boolean, true ONLY if the action involves spending/moving money like pay, cancel, refund, partial_refund, payout, split_payment, create_qr, create_payment_link. MUST be false for read-only actions like track, track_refund, check_settlement, search, faq, remove, handle_failed_payment, magic_checkout_address, check_offers)

Examples:
User: "what is the status of order 123"
Result: {{ "action": "track", "order_id": "123", "requires_confirmation": false }}
User: "did my money settle yet?"
Result: {{ "action": "check_settlement", "requires_confirmation": false }}
User: "remove the black jacket from my cart"
Result: {{ "action": "remove", "item": "black jacket", "requires_confirmation": false }}

User: "add a black jacket to my cart"
Result: {{ "action": "add", "item": "black jacket" }}

User: "I want to checkout the black jacket"
Result: {{ "action": "checkout", "item": "black jacket", "requires_confirmation": true }}
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

User: "refund 500 rupees from order 123"
Result: {{ "action": "partial_refund", "order_id": "123", "amount": 500.0, "requires_confirmation": true }}

User: "my payment failed"
Result: {{ "action": "handle_failed_payment", "requires_confirmation": false }}

User: "get my magic checkout address"
Result: {{ "action": "magic_checkout_address", "requires_confirmation": false }}

User: "what are the current offers"
Result: {{ "action": "check_offers", "requires_confirmation": false }}

User: "create a payment link for 1500.50"
Result: {{ "action": "create_payment_link", "amount": 1500.50, "requires_confirmation": true }}

User: "cancel subscription sub_999"
Result: {{ "action": "cancel_subscription", "sub_id": "sub_999", "requires_confirmation": true }}

User: "what are the charges for UPI?"
Result: {{ "action": "faq", "topic": "upi charges" }}

User: "yes please generate the link"
Result: {{ "action": "confirm_cart", "requires_confirmation": false }}

User: "मुझे नॉइज़ कलर फिट प्रो फोर चाहिए"
Result: {{ "action": "add", "item": "noise color fit pro 4", "requires_confirmation": false }}

User: "क्या आपके पास ब्लैक जैकेट है?"
Result: {{ "action": "search", "item": "black jacket", "requires_confirmation": false }}

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

User: "Track the refund status of pay_1234567890"
Result: {{ "action": "track_refund", "order_id": "pay_1234567890" }}

User: "वोट एयरडोप्स ऐड करना है"
Result: {{ "action": "add", "item": "boat airdopes", "requires_confirmation": false }}

Conversation Context (Recent messages):
{context if context else 'None'}

User: "{text}"
Result:"""
        print(f"Extracting intent for text: '{text}'...")
        response = self.llm(
            prompt,
            max_tokens=256,
            stop=["\n"],
            echo=False,
            temperature=0.1,
            grammar=self.grammar
        )
        
        raw = response['choices'][0]['text']
        start = raw.find('{')
        end = raw.rfind('}')
        
        if start != -1 and end != -1 and end >= start:
            output_text = raw[start:end+1]
        elif start != -1:
            # Handle truncated JSON
            output_text = raw[start:]
            if not output_text.rstrip().endswith("}"):
                output_text += "}"
        else:
            output_text = raw.strip()
            
        try:
            import json
            intent = json.loads(output_text)
            return self._enforce_intent_rules(intent)
        except Exception as e:
            print(f"Failed to parse JSON from Sarvam: {output_text} due to {e}")
            return {"action": "unknown", "item": "unknown", "raw_output": output_text}

intent_pipeline = SarvamIntentPipeline()

if __name__ == "__main__":
    pass
