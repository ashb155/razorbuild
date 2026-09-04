from typing import Optional

SYSTEM_INTENT_PROMPT = """Extract the user intent as a JSON object. Include the following fields when applicable:
- action (search, add, checkout, pay, remove, track, cancel, refund, partial_refund, create_payment_link, cancel_subscription, check_settlement, faq, create_offer, check_emi, create_invoice, payout, split_payment, create_qr, save_card, track_refund, handle_failed_payment, magic_checkout_address, check_offers, confirm_cart)
- item (the product or entity mentioned)
- quantity (default 1 if not specified)
- size (null if not mentioned)
- order_id (for payments, tracking, cancel, refunds)
- amount (float, for payment links, partial refunds, invoices, payouts, or QR codes)
- sub_id (for subscriptions)
- topic (for faqs)
- discount_percent (integer, for smart offers)
- company (string, for invoices)
- recipient (string, for payouts or split payments)
- split_percent (integer, for split payments)
- card_network (string, for saving cards)
- requires_confirmation (boolean, true ONLY for money movement actions: pay, cancel, refund, partial_refund, payout, split_payment, create_qr, create_payment_link. MUST be false for search, track, track_refund, check_settlement, faq, remove, add, handle_failed_payment, magic_checkout_address, check_offers)

Examples:
User: "what is the price of white sneakers"
Result: {{ "action": "search", "item": "white sneakers", "requires_confirmation": false }}

User: "add a black jacket to my cart"
Result: {{ "action": "add", "item": "black jacket", "quantity": 1, "requires_confirmation": false }}

User: "remove the laptop from my cart"
Result: {{ "action": "remove", "item": "laptop", "requires_confirmation": false }}

User: "I want to checkout my cart"
Result: {{ "action": "checkout", "item": "cart", "requires_confirmation": true }}

User: "what is the status of my order"
Result: {{ "action": "track", "requires_confirmation": false }}

User: "create a payment link for 1500.50"
Result: {{ "action": "create_payment_link", "amount": 1500.50, "requires_confirmation": true }}

User: "did my money settle yet?"
Result: {{ "action": "check_settlement", "requires_confirmation": false }}

User: "send an invoice of 5000 to Acme Corp"
Result: {{ "action": "create_invoice", "amount": 5000.0, "company": "Acme Corp", "requires_confirmation": true }}

User: "Pay my employee John Doe 50000 rupees"
Result: {{ "action": "payout", "amount": 50000.0, "recipient": "John Doe", "requires_confirmation": true }}

User: "Split order order_XYZ and send 20 percent to vendor Y"
Result: {{ "action": "split_payment", "order_id": "order_XYZ", "split_percent": 20, "recipient": "vendor Y", "requires_confirmation": true }}

User: "Create a BharatQR code for 500 rupees"
Result: {{ "action": "create_qr", "amount": 500.0, "requires_confirmation": true }}

User: "Save my HDFC credit card for future purchases"
Result: {{ "action": "save_card", "card_network": "HDFC", "requires_confirmation": false }}

User: "Track the refund status of my last order"
Result: {{ "action": "track_refund", "requires_confirmation": false }}

User: "can I pay for this in EMIs?"
Result: {{ "action": "check_emi", "requires_confirmation": false }}

User: "what are the charges for UPI?"
Result: {{ "action": "faq", "topic": "upi charges", "requires_confirmation": false }}

User: "yes please generate the link"
Result: {{ "action": "confirm_cart", "requires_confirmation": false }}

User: "हाँ वो अड कर दो"
Result: {{ "action": "add", "item": "that", "requires_confirmation": false }}

User: "हाँ अड कर दो"
Result: {{ "action": "add", "item": "that", "requires_confirmation": false }}

User: "हा कर दो"
Result: {{ "action": "add", "item": "that", "requires_confirmation": false }}

Conversation Context (Recent messages):
{context}

User: "{text}"
Result:"""

def build_intent_prompt(text: str, context: Optional[str] = None) -> str:
    ctx = context.strip() if context and context.strip() else "None"
    return SYSTEM_INTENT_PROMPT.format(context=ctx, text=text.strip())
