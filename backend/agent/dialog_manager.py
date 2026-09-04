import logging
from typing import Dict, Any, Optional
from models.schemas import ExtractedIntent, AgentResponse
from services.inventory_service import InventoryService
from services.session_service import SessionManager, UserSession
from services.razorpay_service import RazorpayService
from ai.intent_pipeline import SarvamIntentPipeline
from .handlers import CatalogHandler, CartHandler, CustomerServiceHandler, MerchantHandler

logger = logging.getLogger(__name__)

class DialogManager:
    """Orchestrates conversational flow, intent resolution, OTP gating, and action dispatching."""

    NEGATIVE_WORDS = ["no", "nah", "nahi", "mat", "nope", "नहीं", "नही", "मत", "ಬೇಡ"]

    def __init__(
        self,
        inventory_service: InventoryService,
        session_manager: SessionManager,
        razorpay_service: RazorpayService,
        intent_pipeline: SarvamIntentPipeline
    ):
        self.inventory = inventory_service
        self.session_manager = session_manager
        self.razorpay = razorpay_service
        self.intent_pipeline = intent_pipeline
        
        # Initialize handlers
        self.catalog_handler = CatalogHandler(self.inventory)
        self.cart_handler = CartHandler(self.inventory, self.razorpay)
        self.cs_handler = CustomerServiceHandler(self.razorpay, self.inventory)
        self.merchant_handler = MerchantHandler(self.razorpay)

    def process_utterance(
        self,
        query: str,
        phone_number: Optional[str] = None,
        skip_gate: bool = False,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main entry point for processing a voice command or text utterance."""
        session = self.session_manager.get_session(phone_number)
        
        # 1. Confirmation Replay (if user just passed the OTP Confirmation Gate)
        if skip_gate and session.pending_confirmation:
            intent_data = session.pending_confirmation
            session.pending_confirmation = None
            intent = ExtractedIntent(**intent_data)
            logger.info(f"Replaying stored gated intent: {intent}")
        else:
            # Build conversation context for LLM
            llm_context = context or ""
            if session.pending_cross_sell:
                prod = self.inventory.get_product(session.pending_cross_sell)
                name = prod['name'] if prod else session.pending_cross_sell
                llm_context = f"agent: Would you like to add {name} to your order?\n" + llm_context
            elif session.pending_alternative:
                prod = self.inventory.get_product(session.pending_alternative)
                name = prod['name'] if prod else session.pending_alternative
                llm_context = f"agent: We have {name} in stock instead. Should I add that?\n" + llm_context
            elif session.pending_search:
                prod = self.inventory.get_product(session.pending_search)
                name = prod['name'] if prod else session.pending_search
                llm_context = f"agent: The {name} is available. Would you like me to add it to your cart?\n" + llm_context

            # Extract intent using NLU
            raw_intent = self.intent_pipeline.extract_intent(query, context=llm_context)
            intent = ExtractedIntent(**raw_intent)
            
            # Coreference & Dialog State Resolution
            intent = self._resolve_dialog_state(intent, query, session)

        # 2. Check Gated Money Actions (Razorpay Thirdwatch & User OTP Confirmation)
        if self._requires_confirmation_gate(intent, session, skip_gate):
            # Check fraud rate limiter
            if session.is_rate_limited(max_requests=15, window_seconds=60):
                msg = "Razorpay Thirdwatch Alert: Too many financial requests in a short time. Please try again later."
                return AgentResponse(status="blocked", reason="rate_limit_exceeded", message=msg).to_dict()

            session.pending_confirmation = intent.model_dump()
            
            # Format custom confirmation message for full cart checkout
            is_cart_checkout = intent.action in ['checkout', 'buy', 'pay'] and (intent.item == 'cart' or (not intent.item and not intent.order_id))
            if is_cart_checkout:
                if not session.cart:
                    session.pending_confirmation = None
                    return AgentResponse(status="failed", reason="empty_cart", message="Your cart is currently empty. Would you like to search for some products?").to_dict()
                
                inv = self.inventory.get_raw_inventory()
                items_summary = [f"{q}x {inv[k]['name']}" for k, q in session.cart.items() if k in inv]
                total = sum(inv[k]['price'] * q for k, q in session.cart.items() if k in inv)
                msg = f"Your cart has {', '.join(items_summary)}. Total: ₹{total}. Confirm to generate your secure Razorpay payment link?"
            else:
                msg = "This action requires money movement. Initiating OTP Gate... Please confirm before we trigger the Razorpay API."
                
            return AgentResponse(status="pending_confirmation", message=msg).to_dict()

        # 3. Route to Modular Handlers
        response = self._dispatch_intent(intent, session, skip_gate, query)
        return response.to_dict()

    def _resolve_dialog_state(self, intent: ExtractedIntent, query: str, session: UserSession) -> ExtractedIntent:
        """Resolves coreferences (e.g. 'yes', 'add that', 'हा कर दो') against active pending states."""
        is_no = any(w in query.lower() for w in self.NEGATIVE_WORDS)
        matched_item = self.inventory.find_item_in_inventory(query)

        # 1. Pending Cross Sell Resolution
        if session.pending_cross_sell:
            target = session.pending_cross_sell
            session.pending_cross_sell = None
            if is_no:
                intent.action = "decline_suggestion"
                intent.item = None
            elif not matched_item or matched_item == target:
                intent.action = "add"
                intent.item = target
                intent.quantity = 1

        # 2. Pending Alternative Resolution
        elif session.pending_alternative:
            target = session.pending_alternative
            session.pending_alternative = None
            if is_no:
                intent.action = "decline_suggestion"
                intent.item = None
            elif not matched_item or matched_item == target:
                intent.action = "add"
                intent.item = target
                intent.quantity = 1

        # 3. Pending Search Follow-Up Resolution
        elif session.pending_search:
            target = session.pending_search
            session.pending_search = None
            if is_no:
                intent.action = "decline_suggestion"
                intent.item = None
            elif not matched_item or matched_item == target:
                intent.action = "add"
                intent.item = target
                intent.quantity = 1

        return intent

    def _requires_confirmation_gate(self, intent: ExtractedIntent, session: UserSession, skip_gate: bool) -> bool:
        if skip_gate:
            return False
            
        action = intent.action
        item = (intent.item or '').lower()
        
        # Read-only actions never require OTP gate
        if action in ['search', 'price_check', 'find', 'query', 'track', 'track_refund', 'check_settlement', 'faq', 'remove', 'check_emi', 'handle_failed_payment', 'magic_checkout_address', 'check_offers', 'add', 'add_to_cart', 'save_card', 'confirm_cart']:
            return False
            
        # Cart checkout forces confirmation gate
        if action in ['checkout', 'buy', 'pay'] and (not item or item == 'cart'):
            return True
            
        return intent.requires_confirmation

    def _dispatch_intent(self, intent: ExtractedIntent, session: UserSession, skip_gate: bool, raw_query: str) -> AgentResponse:
        action = intent.action
        item = intent.item

        if skip_gate and action in ['checkout', 'buy', 'pay'] and (not item or item.lower() == 'cart'):
            action = 'confirm_cart'

        # 1. Search / Price Check
        if action in ['search', 'price_check', 'find', 'query'] and item:
            return self.catalog_handler.handle_search(intent, session)

        # 2. Cart Checkout & Confirmation
        elif action == 'confirm_cart' or (action in ['checkout', 'buy', 'pay'] and (not item or item.lower() == 'cart')):
            return self.cart_handler.handle_confirm_cart(session)

        # 3. Add to Cart / Purchase Item
        elif action in ['add', 'checkout', 'buy', 'purchase', 'add_to_cart'] and item:
            return self.cart_handler.handle_add(intent, session, skip_gate=skip_gate)

        # 4. Remove from Cart
        elif action in ['remove', 'delete', 'remove_from_cart', 'remove_item'] and item:
            return self.cart_handler.handle_remove(intent, session, raw_query)

        # 5. Order Tracking
        elif action in ['track', 'check_status', 'order_status']:
            return self.cs_handler.handle_track(intent, session)

        # 6. Order Cancellation
        elif action in ['cancel', 'cancel_order']:
            return self.cs_handler.handle_cancel_order(intent, session)

        # 7. Customer Refunds
        elif action in ['refund', 'get_refund']:
            return self.cs_handler.handle_refund(intent, session)
        elif action == 'partial_refund':
            return self.cs_handler.handle_refund(intent, session, amount_inr=intent.amount)
        elif action == 'track_refund':
            return self.cs_handler.handle_track_refund(intent, session)

        # 8. Subscription Management
        elif action in ['cancel_subscription', 'stop_subscription']:
            return self.cs_handler.handle_cancel_subscription(intent, session)

        # 9. EMI & Offers
        elif action in ['check_emi', 'emi', 'affordability']:
            return self.cs_handler.handle_check_emi(intent, session)
        elif action == 'check_offers':
            return self.cs_handler.handle_check_offers(session)
        elif action == 'save_card':
            network = intent.card_network or 'Card'
            session.saved_cards.append(network)
            return AgentResponse(status="success", action="save_card", message=f"Your {network} card has been securely tokenized and saved via Razorpay TokenHQ for future purchases.")

        # 10. Magic Checkout & Failed Payments
        elif action == 'magic_checkout_address':
            msg = f"Found saved Magic Checkout address: 123 Tech Park, Bangalore. Should I use this?"
            return AgentResponse(status="success", action="magic_checkout_address", message=msg)
        elif action == 'handle_failed_payment':
            msg = "Your last payment failed, likely due to a bank timeout. Would you like me to send a new payment link to your saved HDFC card?"
            return AgentResponse(status="success", action="retry_payment", message=msg)

        # 11. Merchant Operations
        elif action in ['create_payment_link', 'payment_link', 'generate_payment_link']:
            return self.merchant_handler.handle_payment_link(intent, session)
        elif action in ['check_settlement', 'settlement_status', 'check_settlements']:
            return self.merchant_handler.handle_settlements()
        elif action in ['create_offer', 'create_discount', 'offer', 'discount']:
            return self.merchant_handler.handle_create_offer(intent, session)
        elif action in ['create_invoice', 'invoice', 'generate_invoice']:
            return self.merchant_handler.handle_invoice(intent, session)
        elif action in ['payout', 'send_payout', 'transfer']:
            return self.merchant_handler.handle_payout(intent, session)
        elif action in ['split_payment', 'split', 'route']:
            return self.merchant_handler.handle_split_payment(intent)
        elif action == 'create_qr':
            return self.merchant_handler.handle_create_qr(intent)

        # 12. Decline Suggestion / Offer
        elif action in ['decline_suggestion', 'decline_offer']:
            return AgentResponse(status="success", action="decline_suggestion", message="No problem! Let me know if you need anything else, or say 'checkout' when you're ready.")

        # 13. Support & FAQs
        elif action in ['faq', 'question', 'help', 'support']:
            topic = intent.topic or "general"
            return AgentResponse(status="routed_to_support", message=f"It looks like you have a question about '{topic}'. Let me connect you with our knowledge base or a support agent.")

        # 14. Unhandled Fallback
        return AgentResponse(status="failed", reason="unhandled_intent", message=f"I'm sorry, I couldn't quite understand what you wanted to do with '{item or 'that'}'.")
