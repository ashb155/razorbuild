import os
import json
import logging
from typing import Dict, Any, Optional
from models.schemas import ExtractedIntent
from .prompts import build_intent_prompt

logger = logging.getLogger(__name__)

class SarvamIntentPipeline:
    """NLU Intent extraction engine powered by Sarvam-1 local LLM with grammar constraints."""

    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sarvam-1.Q4_K_M.gguf")
        self.llm = None
        self.grammar = None
        self._load_model()

    def _load_model(self):
        try:
            from llama_cpp import Llama, LlamaGrammar
            if os.path.exists(self.model_path):
                logger.info(f"Loading Sarvam-1 GGUF from {self.model_path}...")
                self.llm = Llama(model_path=self.model_path, n_ctx=4096, verbose=False)
                
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
                logger.info("Successfully loaded Sarvam-1 LLM with JSON grammar.")
            else:
                logger.warning(f"Sarvam-1 model file not found at {self.model_path}")
        except ImportError:
            logger.warning("llama-cpp-python is not installed, running in offline fallback mode.")

    def extract_intent(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Extracts structured intent from user utterance."""
        if not self.llm:
            return {"action": "unknown", "item": None, "requires_confirmation": False}

        prompt = build_intent_prompt(text, context)
        try:
            response = self.llm(
                prompt,
                max_tokens=256,
                stop=["\n"],
                echo=False,
                temperature=0.1,
                grammar=self.grammar
            )
            raw_text = response['choices'][0]['text']
            
            # Robust JSON extraction
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start != -1 and end != -1 and end >= start:
                json_str = raw_text[start:end+1]
            elif start != -1:
                json_str = raw_text[start:] + ("}" if not raw_text.rstrip().endswith("}") else "")
            else:
                json_str = raw_text.strip()
                
            data = json.loads(json_str)
            intent = ExtractedIntent(
                action=data.get('action', 'unknown'),
                item=data.get('item'),
                quantity=data.get('quantity', 1) or 1,
                size=data.get('size'),
                order_id=data.get('order_id'),
                amount=data.get('amount'),
                sub_id=data.get('sub_id'),
                topic=data.get('topic'),
                discount_percent=data.get('discount_percent'),
                company=data.get('company'),
                recipient=data.get('recipient'),
                split_percent=data.get('split_percent'),
                card_network=data.get('card_network'),
                requires_confirmation=data.get('action') in [
                    'pay', 'refund', 'partial_refund', 'create_payment_link',
                    'cancel_subscription', 'create_offer', 'create_invoice',
                    'payout', 'split_payment', 'create_qr'
                ]
            )
            return intent.dict()
        except Exception as e:
            logger.error(f"Intent extraction parsing error: {e}")
            return {"action": "unknown", "item": None, "requires_confirmation": False}

intent_pipeline = SarvamIntentPipeline()
