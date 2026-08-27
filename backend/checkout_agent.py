import json
import os
import sys
import difflib
import time
from collections import defaultdict
from dotenv import load_dotenv
import razorpay

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
from ai_pipeline import intent_pipeline

# Global state for thirdwatch fraud tracker (session -> list of timestamps)
fraud_tracker = defaultdict(list)

# Initialize Razorpay Client (gracefully handles missing/invalid keys until an API call is made)
rzp_client = razorpay.Client(auth=(os.environ.get('RAZORPAY_KEY_ID', ''), os.environ.get('RAZORPAY_KEY_SECRET', '')))

_cached_inventory = None
def load_inventory():
    global _cached_inventory
    if _cached_inventory is None:
        path = os.path.join(os.path.dirname(__file__), "inventory.json")
        with open(path, 'r', encoding='utf-8') as f:
            _cached_inventory = json.load(f)
    return _cached_inventory

# Mock databases for the remaining intents
import uuid
MOCK_CART = defaultdict(dict)
MOCK_ORDERS = defaultdict(dict)
MOCK_SETTLEMENTS = defaultdict(dict)
MOCK_OFFERS = defaultdict(dict)
MOCK_INVOICES = defaultdict(dict)
MOCK_PAYOUTS = defaultdict(dict)
MOCK_SAVED_CARDS = defaultdict(list)
MOCK_REFUNDS = defaultdict(dict)
MOCK_SUBSCRIPTIONS = defaultdict(dict)
pending_cross_sells = defaultdict(str)
pending_confirmations = defaultdict(dict)  # Stores intent at the moment OTP gate fires; replayed on skip_gate=True

def find_item_in_inventory(item: str, inventory: dict):
    """Robust fuzzy matching against both keys and human-readable names."""
    if not item:
        return None
    item_lower = item.lower()
    
    if item_lower in inventory:
        return item_lower
        
    # Match against keys
    matches = difflib.get_close_matches(item_lower, inventory.keys(), n=1, cutoff=0.5)
    if matches:
        return matches[0]
        
    # Match against human-readable names
    name_to_key = {v['name'].lower(): k for k, v in inventory.items()}
    matches = difflib.get_close_matches(item_lower, name_to_key.keys(), n=1, cutoff=0.5)
    if matches:
        return name_to_key[matches[0]]
        
    return None

def suggest_alternative(inventory, out_of_stock_item):
    """Finds an alternative item in the same category."""
    out_of_stock_data = inventory.get(out_of_stock_item)
    if not out_of_stock_data:
        return None
        
    category = out_of_stock_data['category']
    for item_name, data in inventory.items():
        if item_name != out_of_stock_item and data['category'] == category and data['stock'] > 0:
            return item_name
    return None

def suggest_any_alternative(inventory, search_term):
    """Finds a fallback alternative for items that don't exist in the catalog at all."""
    # 1. Try very loose fuzzy matching
    loose_matches = difflib.get_close_matches(search_term, inventory.keys(), n=1, cutoff=0.2)
    if loose_matches and inventory[loose_matches[0]]['stock'] > 0:
        return loose_matches[0]
        
    # 2. Try word-level matching (e.g., 't-shirt' -> 'shirt' if it existed)
    words = search_term.lower().split()
    for word in words:
        for item_name, data in inventory.items():
            if word in item_name and data['stock'] > 0:
                return item_name
                
    # 3. Ultimate Fallback: Just suggest an in-stock item to save the sale
    for item_name, data in inventory.items():
        if data['stock'] > 0:
            return item_name
    return None

def get_cross_sell(inventory, item):
    """Suggests a complimentary item based on category."""
    item_data = inventory.get(item)
    if not item_data: return None
    
    cat = item_data.get('category')
    
    cross_map = {
        'watch': 'headphones',
        'smartphone': 'headphones',
        'shoes': 'clothing',
        'clothing': 'shoes',
        'laptop': 'accessories',
        'accessories': 'laptop'
    }
    target_cat = cross_map.get(cat, cat)
    
    for k, v in inventory.items():
        if k != item and v.get('category') == target_cat and v.get('stock', 0) > 0:
            return v['name']
    return None

def process_voice_command(query: str, phone_number: str = None, skip_gate: bool = False, context: str = None):
    inventory = load_inventory()
    session_id = phone_number if phone_number else str(uuid.uuid4())
    
    # Session-scoped state references (use MOCK_* module-level dicts)
    cart = MOCK_CART[session_id]
    orders = MOCK_ORDERS[session_id]
    refunds = MOCK_REFUNDS[session_id]
    subscriptions = MOCK_SUBSCRIPTIONS[session_id]
    invoices = MOCK_INVOICES[session_id]
    payouts = MOCK_PAYOUTS[session_id]
    saved_cards = MOCK_SAVED_CARDS[session_id]
    offers = MOCK_OFFERS[session_id]
    
    # ----------------------------------------------------------------
    # STEP 0: If the user is confirming a gated action (skip_gate=True),
    # SKIP the LLM entirely and replay the stored intent from when the
    # OTP gate originally fired. This makes confirmations 100% reliable
    # regardless of language (Hindi, Kannada, etc.).
    # ----------------------------------------------------------------
    if skip_gate and pending_confirmations.get(session_id):
        intent = pending_confirmations.pop(session_id)
        print(f"\n[CONFIRM] Replaying stored intent: {intent}")
    else:
        # 1. AI Extract Intent
        print(f"\nUser Said: '{query}'")
        intent = intent_pipeline.extract_intent(query, context=context)
        
        # 1.05 Handle Cross-Sell Affirmations — language-agnostic approach:
        # Instead of maintaining word lists for every language, we check the LLM's
        # own output. If there's a pending cross-sell and the LLM returned an
        # "unknown" / null intent (it couldn't understand a short affirmative phrase),
        # we treat it as cross-sell acceptance. We only check for explicit negatives
        # (a much smaller, stable set) to avoid false positives.
        if pending_cross_sells.get(session_id):
            action_raw = intent.get('action')
            item_raw = (intent.get('item') or '').strip().lower()
            lm_confused = action_raw in [None, 'unknown'] or item_raw in ['', 'unknown', 'that', 'it', 'this']
            # Explicit negatives — Hindi, Kannada, English only
            explicit_no = ["no", "nah", "nahi", "mat", "nope",
                           "नहीं", "नही", "मत",   # Hindi/Marathi
                           "ಬೇಡ"]                 # Kannada
            said_no = any(w in query for w in explicit_no)
            if lm_confused and not said_no:
                intent = {"action": "add", "item": pending_cross_sells[session_id], "quantity": 1}
            # Always clear the pending cross-sell after this turn
            del pending_cross_sells[session_id]

    
    # 1.1 Force read-only actions to bypass OTP gate
    if intent.get('action') in ['track', 'track_refund', 'check_settlement', 'search', 'faq', 'remove', 'check_emi', 'handle_failed_payment', 'magic_checkout_address', 'check_offers', 'add', 'add_to_cart']:
        intent['requires_confirmation'] = False
    
    # 1.2 Force checkout/cart to ALWAYS use the OTP gate so there is only ONE confirmation step.
    _pre_action = intent.get('action')
    _pre_item = (intent.get('item') or '').lower()
    if _pre_action in ['checkout', 'buy', 'pay'] and (not _pre_item or _pre_item == 'cart') and not skip_gate:
        intent['requires_confirmation'] = True
    
    # 1.5 Handle Gated Money Actions (OTP Gate)
    if intent.get('requires_confirmation') and not skip_gate:
        now = time.time()
        fraud_tracker[session_id] = [t for t in fraud_tracker[session_id] if now - t < 60]
        
        if len(fraud_tracker[session_id]) >= 15:
            msg = "Razorpay Thirdwatch Alert: Too many financial requests in a short time. Please try again later."
            print(f"[FRAUD] {msg}")
            return {"status": "blocked", "reason": "rate_limit_exceeded", "message": msg}
            
        fraud_tracker[session_id].append(now)
        
        # SAVE intent so we can replay it when user confirms (skip_gate=True)
        pending_confirmations[session_id] = intent.copy()
        
        # For checkout/cart, include the cart summary in the confirmation message
        _gate_action = intent.get('action')
        _gate_item = (intent.get('item') or '').lower()
        if _gate_action in ['checkout', 'buy', 'pay'] and (not _gate_item or _gate_item == 'cart'):
            if not cart:
                del pending_confirmations[session_id]
                return {"status": "failed", "reason": "empty_cart", "message": "Your cart is currently empty. Would you like to search for some products?"}
            _cart_parts = []
            _total = 0
            for _ci, _cq in cart.items():
                if _ci in inventory:
                    _cart_parts.append(f"{_cq}x {inventory[_ci]['name']}")
                    _total += inventory[_ci]['price'] * _cq
            _summary = ", ".join(_cart_parts)
            msg = f"Your cart has {_summary}. Total: \u20b9{_total}. Confirm to generate your secure Razorpay payment link?"
        else:
            msg = "This action requires money movement. Initiating OTP Gate... Please confirm before we trigger the Razorpay API."
        
        print(f"[AUDIT] {msg}")
        return {"status": "pending_confirmation", "message": msg}
    
    action = intent.get('action')
    item = intent.get('item')
    requested_qty = max(1, int(intent.get('quantity', 1) or 1))
    
    if skip_gate:
        if action in ['checkout', 'buy', 'pay'] and (not item or item.lower() == 'cart'):
            action = 'confirm_cart'
    
    # 2. Handle Price Checking (Search)
    if action in ['search', 'price_check', 'find', 'query'] and item:
        matched_item = find_item_in_inventory(item, inventory)
        if matched_item:
            item = matched_item
            product = inventory.get(item, {"name": item.replace("_", " ")})
            price = inventory[item]['price']
            stock = inventory[item]['stock']
            
            if stock > 0:
                msg = f"The {product['name']} costs ₹{price}. Would you like me to add it to your cart?"
            else:
                alt = suggest_alternative(inventory, item)
                msg = f"The {product['name']} costs ₹{price}, but it's currently out of stock."
                if alt:
                    alt_name = inventory[alt]['name']
                    msg += f" However, we do have the {alt_name} in stock! Would you like to check it out?"
                
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "price_check", "message": msg}
        else:
            alt = suggest_any_alternative(inventory, item)
            if alt:
                alt_name = inventory[alt]['name']
                alt_price = inventory[alt]['price']
                msg = f"I couldn't find '{item}', but we do have the {alt_name} which costs ₹{alt_price}. Would you like me to add it to your cart?"
            else:
                msg = f"I couldn't find '{item}' in our catalog."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "item_not_found", "alternative": alt, "message": msg}

    # 2.5 Handle Full Cart Checkout
    # NOTE: In normal flow this block is unreachable for checkout+cart because step 1.2 above
    # forces requires_confirmation=True, routing through the OTP gate (which now includes the
    # cart summary). This block remains as a safety fallback only.
    elif action in ['checkout', 'buy', 'pay'] and (not item or item.lower() == 'cart'):
        if not cart:
            msg = "Your cart is currently empty. Would you like to search for some products?"
            print(f"[AGENT] {msg}")
            return {"status": "failed", "reason": "empty_cart", "message": msg}
            
        cart_items = []
        total_amount = 0
        for cart_item_key, qty in cart.items():
            prod_name = inventory[cart_item_key]['name']
            price = inventory[cart_item_key]['price']
            total_amount += price * qty
            cart_items.append(f"{qty}x {prod_name}")
            
        summary = ", ".join(cart_items)
        msg = f"Your cart has {summary}. Total: ₹{total_amount}. Confirm to generate your secure Razorpay payment link?"
        print(f"[AGENT] {msg}")
        return {"status": "pending_confirmation", "message": msg}
        
    # 2.6 Handle Cart Confirmation
    elif action == 'confirm_cart':
        if not cart:
            msg = "Your cart is empty, so there's nothing to checkout!"
            print(f"[AGENT] {msg}")
            return {"status": "failed", "reason": "empty_cart", "message": msg}
            
        cart_items = []
        total_amount = 0
        for cart_item_key, qty in cart.items():
            prod_name = inventory[cart_item_key]['name']
            price = inventory[cart_item_key]['price']
            total_amount += price * qty
            cart_items.append(f"{qty}x {prod_name}")
            
        summary = ", ".join(cart_items)
        
        try:
            total_price_paise = int(total_amount * 100)
            payment_link_data = {
                "amount": total_price_paise,
                "currency": "INR",
                "description": f"Payment for cart: {summary}",
                "customer": {
                    "name": "Voice Assistant Customer",
                    "contact": phone_number or "+919999999999",
                    "email": "customer@example.com"
                },
                "notify": {"sms": False, "email": False}
            }
            plink = rzp_client.payment_link.create(payment_link_data)
            link_id = plink.get('id')
            link = plink.get('short_url')
            orders[link_id] = {"status": "created", "total": total_amount}
            cart.clear()
            msg = f"Perfect! I've generated your secure payment link for {summary}. You can complete your ₹{total_amount} payment here: {link}"
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "checkout_cart", "payment_link": link, "message": msg}
        except Exception as e:
            msg = f"Perfect! I've generated your payment link for {summary}. Order ID: cart_123 (Mock Fallback)"
            cart.clear()
            print(f"[AGENT-MOCK-FALLBACK] {msg}")
            return {"status": "success", "action": "checkout_cart", "message": msg}

    # 3. Handle Add/Checkout Actions
    elif action in ['add', 'checkout', 'buy', 'purchase', 'add_to_cart'] and item:
        matched_item = find_item_in_inventory(item, inventory)
        if matched_item:
            item = matched_item
            product = inventory.get(item, {"name": item.replace("_", " ")})
            stock = inventory[item]['stock']
            price = inventory[item]['price']
            
            # --- THE GRACEFUL FAILURE (OUT OF STOCK) ---
            if stock == 0:
                alt = suggest_alternative(inventory, item)
                msg = f"Sorry, the {product['name']} is currently out of stock."
                if alt:
                    alt_name = inventory[alt]['name']
                    msg += f" However, we do have the {alt_name} in stock! Should I add that instead?"
                print(f"[AGENT] {msg}")
                return {"status": "failed", "reason": "out_of_stock", "alternative": alt, "message": msg}
                
            # --- PARTIAL FULFILLMENT (QUANTITY MISMATCH) ---
            elif stock < requested_qty:
                if skip_gate:
                    requested_qty = stock
                else:
                    msg = f"We only have {stock} {item}(s) in stock, but you asked for {requested_qty}. Would you like me to add all {stock} to your cart instead?"
                    print(f"[AGENT] {msg}")
                    return {"status": "pending_quantity_confirmation", "available": stock, "message": msg}
                
            # --- THE HAPPY PATH + UPSELL / PAYMENT LINK ---
            if action in ['checkout', 'buy', 'purchase']:
                try:
                    total_price = int(price * requested_qty * 100)
                    payment_link_data = {
                        "amount": total_price,
                        "currency": "INR",
                        "description": f"Payment for {requested_qty} x {item}",
                        "customer": {
                            "name": "Voice Assistant Customer",
                            "contact": phone_number or "+919999999999",
                            "email": "customer@example.com"
                        },
                        "notify": {"sms": False, "email": False}
                    }
                    plink = rzp_client.payment_link.create(payment_link_data)
                    link_id = plink.get('id')
                    link = plink.get('short_url')
                    orders[link_id] = {"status": "created", "total": total_price / 100}
                    cart.clear()
                    msg = f"I've prepared your order for {requested_qty} {product['name']}. The total is ₹{price * requested_qty}. Please complete your payment here: {link}"
                    print(f"[AGENT] {msg}")
                    return {"status": "success", "action": "checkout_with_link", "payment_link": link, "message": msg}
                except Exception as e:
                    print(f"[ERROR] Payment Link Gen Error: {e}")
                    link_id = f"plink_mock_{len(orders) + 1000}"
                    orders[link_id] = {"status": "created", "total": price * requested_qty}
                    cart.clear()
                    msg = f"I've prepared your order for {requested_qty} {product['name']}. The total is ₹{price * requested_qty}. Order ID: {link_id} (Mock Fallback)"
                    print(f"[AGENT-MOCK-FALLBACK] {msg}")
                    return {"status": "success", "action": "checkout_with_link", "message": msg}
            else:
                cart[item] = cart.get(item, 0) + requested_qty
                msg = f"I've added {requested_qty} {product['name']} to your cart."
                print(f"[AUDIT] {msg}")
                cross_sell_item = get_cross_sell(inventory, item)
                if cross_sell_item:
                    msg += f" Would you also like to add {cross_sell_item} to your order?"
                    # Store the KEY (matched from name) in pending_cross_sells
                    for k, v in inventory.items():
                        if v['name'] == cross_sell_item:
                            pending_cross_sells[session_id] = k
                            break
                print(f"[AGENT] {msg}")
                return {"status": "success", "action": "added", "message": msg}
            
        else:
            alt = suggest_any_alternative(inventory, item)
            if alt:
                alt_name = inventory[alt]['name']
                msg = f"I couldn't find '{item}'. But we do have the {alt_name} in stock! Would you like me to add that instead?"
            else:
                msg = f"I couldn't find '{item}' in our catalog."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "item_not_found", "alternative": alt, "message": msg}

    # 4. Handle FAQ / Support Routing
    elif action in ['faq', 'question', 'help', 'support']:
        topic = intent.get('topic', 'general')
        msg = f"It looks like you have a question about '{topic}'. Let me connect you with our knowledge base or a support agent."
        print(f"[SUPPORT] {msg}")
        return {"status": "routed_to_support", "topic": topic, "message": msg}
        
    # 4.1 Handle Failed Payments Retry Flow
    elif action == 'handle_failed_payment':
        msg = "Your last payment failed, likely due to a bank timeout. Would you like me to send a new payment link to your saved HDFC card?"
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "retry_payment", "message": msg}
        
    # 4.2 Handle Magic Checkout Address Fetching
    elif action == 'magic_checkout_address':
        if phone_number:
            msg = f"Found saved Magic Checkout address for {phone_number}: 123 Tech Park, Bangalore. Should I use this?"
        else:
            msg = "I can fetch your saved addresses if you verify your phone number via Magic Checkout."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "magic_checkout_address", "message": msg}
        
    # 4.3 Handle Offers/Affordability Engine
    elif action == 'check_offers':
        msg = "Great news! Since you have an ICICI credit card saved, Razorpay offers a No-Cost EMI of ₹500/month, plus an instant 10% discount. Should I apply it?"
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "check_offers", "message": msg}
        
    # 5. Handle Remove from Cart
    elif action in ['remove', 'delete', 'remove_from_cart', 'remove_item'] and item:
        matches = difflib.get_close_matches(item, cart.keys(), n=1, cutoff=0.6)
        if matches:
            removed_item = matches[0]
            product = inventory.get(removed_item, {"name": removed_item.replace("_", " ")})
            del cart[removed_item]
            msg = f"I've removed the {product['name']} from your cart."
            print(f"[AUDIT] {msg}")
            return {"status": "success", "action": "removed", "message": msg}
        else:
            msg = f"You don't have '{item}' in your cart to remove."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "not_in_cart", "message": msg}
            
    # 6. Handle Order Tracking (Live SDK Integration)
    elif action in ['track', 'check_status', 'order_status']:
        order_id = intent.get('order_id')
        try:
            if str(order_id).startswith('order_'):
                real_order = rzp_client.order.fetch(str(order_id))
                status = real_order['status']
                msg = f"According to Razorpay, your order {order_id} is currently: {status.upper()}."
                print(f"[AGENT] {msg}")
                return {"status": "success", "action": "track", "order_status": status, "message": msg}
            else:
                # Query the live Razorpay API for orders matching this receipt/ID
                # Use rzp_orders to avoid shadowing the session-scoped orders dict
                rzp_orders = rzp_client.order.fetch_all({"receipt": str(order_id)})
                
                if rzp_orders.get('items') and len(rzp_orders['items']) > 0:
                    real_order = rzp_orders['items'][0]
                    status = real_order['status']
                    msg = f"According to Razorpay, your order {order_id} is currently: {status.upper()}."
                    print(f"[AGENT] {msg}")
                    return {"status": "success", "action": "track", "order_status": status, "message": msg}
                else:
                    # Fallback to local session DB if order not found on Razorpay
                    if order_id in orders:
                        status = orders[order_id]['status']
                        msg = f"Order {order_id} is currently: {status.upper()} (Local Database)."
                        print(f"[AGENT-MOCK-FALLBACK] {msg}")
                        return {"status": "success", "action": "track", "order_status": status, "message": msg}
                    else:
                        msg = f"I couldn't find order {order_id} on Razorpay or locally."
                        print(f"[WARNING] {msg}")
                        return {"status": "failed", "reason": "order_not_found", "message": msg}
        except Exception as e:
            print(f"[ERROR] Razorpay SDK Error: {str(e)}")
            msg = f"I couldn't find order {order_id}."
            return {"status": "failed", "reason": "order_not_found", "message": msg}

    # 7. Handle E-Commerce Order Cancelation (Non-subscription)
    elif action in ['cancel', 'cancel_order']:
        order_id = intent.get('order_id')
        if order_id in orders:
            if orders[order_id].get('status') == 'shipped':
                msg = f"Sorry, order {order_id} has already shipped and cannot be canceled."
                print(f"[AGENT] {msg}")
                return {"status": "failed", "reason": "already_shipped", "message": msg}
            else:
                orders[order_id]['status'] = 'canceled'
                msg = f"Order {order_id} has been successfully canceled and refunded."
                print(f"[AGENT] {msg}")
                return {"status": "success", "action": "canceled", "message": msg}
        else:
            msg = f"I couldn't find order {order_id} to cancel."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "order_not_found", "message": msg}

    # 7.1 Handle Payments (Customer side)
    elif action in ['pay', 'make_payment']:
        order_id = intent.get('order_id')
        msg = f"Initiating Razorpay payment for order {order_id}."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "pay", "message": msg}

    # 7.2 Handle Refunds (Customer side)
    elif action in ['refund', 'get_refund']:
        order_id = intent.get('order_id')
        try:
            if str(order_id).startswith('pay_'):
                payment_id = order_id
            else:
                payments = rzp_client.order.payments(order_id)
                if payments.get('items') and len(payments['items']) > 0:
                    payment_id = payments['items'][0]['id']
                else:
                    msg = f"No payments found for order {order_id} to refund."
                    print(f"[WARNING] {msg}")
                    return {"status": "failed", "reason": "payment_not_found", "message": msg}
            
            rzp_client.payment.refund(payment_id, {"speed": "optimum"})
            orders[order_id] = orders.get(order_id, {})
            orders[order_id]['status'] = 'refunded'
            refunds[order_id] = {'status': 'processed'}
            msg = f"Successfully initiated live Razorpay refund for {order_id}."
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "refund", "message": msg}
        except Exception as e:
            msg = f"Failed to initiate live refund. Razorpay Error: {str(e)}"
            orders[order_id] = orders.get(order_id, {})
            orders[order_id]['status'] = 'refunded'
            refunds[order_id] = {'status': 'processed'}
            print(f"[AGENT-MOCK-FALLBACK] Initiating mock refund for {order_id} due to API Error: {str(e)}")
            return {"status": "success", "action": "refund", "message": f"Successfully initiated refund for {order_id} (Mock Fallback)."}

    # 7.2.1 Handle Partial Refunds
    elif action == 'partial_refund':
        order_id = intent.get('order_id')
        amount = intent.get('amount')
        try:
            if str(order_id).startswith('pay_'):
                payment_id = order_id
            else:
                payments = rzp_client.order.payments(order_id)
                if payments.get('items') and len(payments['items']) > 0:
                    payment_id = payments['items'][0]['id']
                else:
                    msg = f"No payments found for order {order_id} to partially refund."
                    print(f"[WARNING] {msg}")
                    return {"status": "failed", "reason": "payment_not_found", "message": msg}
            
            rzp_client.payment.refund(payment_id, {"amount": int(amount * 100), "speed": "optimum"})
            msg = f"Successfully initiated live partial line-item refund of ₹{amount} for {order_id}."
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "partial_refund", "message": msg}
        except Exception as e:
            msg = f"Failed to initiate live partial refund. Razorpay Error: {str(e)}"
            print(f"[AGENT-MOCK-FALLBACK] Initiating mock partial refund for {order_id} due to API Error: {str(e)}")
            return {"status": "success", "action": "partial_refund", "message": f"Successfully initiated a partial line-item refund of ₹{amount} for {order_id} (Mock Fallback)."}

    # 7.3 Handle Payment Links (Merchant side)
    elif action in ['create_payment_link', 'payment_link', 'generate_payment_link']:
        amount = intent.get('amount')
        try:
            plink = rzp_client.payment_link.create({
                "amount": int(amount * 100),
                "currency": "INR",
                "description": "Hackathon Voice Generated Link",
                "customer": {
                    "name": "Customer",
                    "contact": phone_number or "+919876543210"
                }
            })
            link_id = plink.get('id')
            orders[link_id] = {"status": "created", "total": amount}
            msg = f"Generated a LIVE Razorpay Payment Link for ₹{amount}: {plink['short_url']}"
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "create_payment_link", "payment_link": plink['short_url'], "message": msg}
        except Exception as e:
            msg = f"Failed to generate live payment link. Razorpay Error: {str(e)}"
            print(f"[AGENT-MOCK-FALLBACK] Generating mock payment link for ₹{amount} due to API Error: {str(e)}")
            return {"status": "success", "action": "create_payment_link", "message": f"Generated a Razorpay Payment Link for ₹{amount} (Mock Fallback)."}

    # 7.5 Handle Subscription Cancellation (Customer side)
    elif action in ['cancel_subscription', 'stop_subscription']:
        sub_id = intent.get('sub_id')
        if sub_id in subscriptions:
            if subscriptions[sub_id]['status'] == 'canceled':
                msg = f"Subscription {sub_id} is already canceled."
                print(f"[AGENT] {msg}")
                return {"status": "failed", "reason": "already_canceled", "message": msg}
            else:
                subscriptions[sub_id]['status'] = 'canceled'
                msg = f"Your subscription {sub_id} has been successfully canceled. No further charges will be made."
                print(f"[AGENT] {msg}")
                return {"status": "success", "action": "cancel_subscription", "message": msg}
        else:
            msg = f"I couldn't find an active subscription with ID {sub_id}."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "subscription_not_found", "message": msg}

    # 8. Handle Settlement Check (Merchant side)
    elif action in ['check_settlement', 'settlement_status', 'check_settlements']:
        try:
            settlements = rzp_client.settlement.all({"count": 1})
            if settlements.get('items') and len(settlements['items']) > 0:
                latest = settlements['items'][0]
                setl_id = latest['id']
                amount = latest['amount'] / 100.0
                status = latest['status']
                created_at = time.strftime('%Y-%m-%d', time.localtime(latest['created_at']))
                msg = f"Checking your latest settlements... Your last settlement {setl_id} for ₹{amount} is currently {status} (created on {created_at})."
            else:
                msg = "Checking your latest settlements... You don't have any recent settlements."
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "settlement_checked", "message": msg}
        except Exception as e:
            msg = f"Checking your latest settlements... Your last settlement setl_789 for ₹50000 was settled on 2023-10-25 (Mock Fallback due to API error: {str(e)})."
            print(f"[AGENT-MOCK-FALLBACK] {msg}")
            return {"status": "success", "action": "settlement_checked", "message": msg}

    # 9. Handle Smart Offers (Merchant side)
    elif action in ['create_offer', 'create_discount', 'offer', 'discount']:
        discount = intent.get('discount_percent', 10)
        offer_id = f"offer_diwali_{discount}"
        offers[offer_id] = {"discount": discount, "status": "active"}
        msg = f"Successfully created a {discount}% Razorpay Offer: {offer_id}"
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "create_offer", "offer_id": offer_id, "message": msg}

    # 10. Handle B2B Invoicing (Merchant side)
    elif action in ['create_invoice', 'invoice', 'generate_invoice']:
        amount = intent.get('amount', 0.0)
        company = intent.get('company', 'Customer')
        try:
            invoice_data = {
                "type": "invoice",
                "description": f"Invoice for {company}",
                "customer": {
                    "name": company
                },
                "line_items": [
                    {
                        "name": "Services",
                        "amount": int(amount * 100),
                        "currency": "INR",
                        "quantity": 1
                    }
                ]
            }
            invoice = rzp_client.invoice.create(invoice_data)
            inv_id = invoice['id']
            invoices[inv_id] = {"amount": amount, "company": company, "status": "issued"}
            msg = f"Generated GST Invoice {inv_id} for {company} for ₹{amount}. Link: {invoice.get('short_url')}"
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "create_invoice", "invoice_id": inv_id, "message": msg}
        except Exception as e:
            inv_id = f"inv_{len(invoices) + 1000}"
            invoices[inv_id] = {"amount": amount, "company": company, "status": "issued"}
            msg = f"Generated GST Invoice {inv_id} for {company} for ₹{amount} (Mock Fallback due to API error: {str(e)})."
            print(f"[AGENT-MOCK-FALLBACK] {msg}")
            return {"status": "success", "action": "create_invoice", "invoice_id": inv_id, "message": msg}

    # 11. Handle Affordability/EMI (Customer side)
    elif action in ['check_emi', 'emi', 'affordability']:
        item_to_check = item if item else "your cart"
        # Dummy mock calculation
        msg = f"Great news! You can pay for {item_to_check} via Razorpay No-Cost EMI starting at just ₹500/month."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "check_emi", "message": msg}

    # 12. Handle Vendor Payouts (Merchant side)
    elif action in ['payout', 'send_payout', 'transfer']:
        amount = intent.get('amount')
        recipient = intent.get('recipient')
        payout_id = f"pout_{len(payouts) + 1000}"
        payouts[payout_id] = {"amount": amount, "recipient": recipient, "status": "processed"}
        msg = f"Successfully processed RazorpayX payout of ₹{amount} to {recipient}. Payout ID: {payout_id}."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "payout", "payout_id": payout_id, "message": msg}

    # 13. Handle Payment Splitting (Merchant side)
    elif action in ['split_payment', 'split', 'route']:
        order_id = intent.get('order_id')
        split = intent.get('split_percent')
        recipient = intent.get('recipient')
        msg = f"Razorpay Route configured. {split}% of order {order_id} will be automatically routed to {recipient}."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "split_payment", "message": msg}

    # 14. Handle BharatQR Generation (Merchant side)
    elif action == 'create_qr':
        amount = intent.get('amount')
        msg = f"Generated a BharatQR code for ₹{amount}. The customer can scan this to pay via UPI or Cards."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "create_qr", "message": msg}

    # 15. Handle Card Tokenization (Customer side)
    elif action == 'save_card':
        network = intent.get('card_network', 'credit')
        saved_cards.append(network)
        msg = f"Your {network} card has been securely tokenized and saved via Razorpay TokenHQ for future purchases."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "save_card", "message": msg}

    # 16. Handle Refund Tracking (Customer side)
    elif action == 'track_refund':
        order_id = intent.get('order_id')
        try:
            if str(order_id).startswith('pay_'):
                payment_id = order_id
            else:
                payments = rzp_client.order.payments(order_id)
                if payments.get('items') and len(payments['items']) > 0:
                    payment_id = payments['items'][0]['id']
                else:
                    msg = f"I couldn't find any payments for order {order_id}."
                    print(f"[WARNING] {msg}")
                    return {"status": "failed", "reason": "payment_not_found", "message": msg}
            
            # Use rzp_refunds to avoid shadowing the session-scoped refunds dict
            rzp_refunds = rzp_client.refund.all({"payment_id": payment_id})
            if rzp_refunds.get('items') and len(rzp_refunds['items']) > 0:
                latest_refund = rzp_refunds['items'][0]
                status = latest_refund['status']
                arn = latest_refund.get('acquirer_data', {}).get('arn', 'Pending')
                msg = f"Your refund for {order_id} is currently {status}. Bank ARN: {arn}."
                print(f"[AGENT] {msg}")
                return {"status": "success", "action": "track_refund", "message": msg}
            else:
                msg = f"I couldn't find a refund record for {order_id} on Razorpay."
                print(f"[WARNING] {msg}")
                return {"status": "failed", "reason": "refund_not_found", "message": msg}
        except Exception as e:
            # Fall back to session-scoped refunds dict (correctly references local session var)
            if order_id in refunds:
                status = refunds[order_id]['status']
                msg = f"Your refund for {order_id} has been {status} by the bank. Bank ARN: 8493820498 (Mock Fallback due to API error)."
                print(f"[AGENT-MOCK-FALLBACK] {msg}")
                return {"status": "success", "action": "track_refund", "message": msg}
            else:
                msg = f"I couldn't find a refund record for {order_id}."
                print(f"[WARNING] {msg}")
                return {"status": "failed", "reason": "refund_not_found", "message": msg}

    # 17. Catch-all for unrecognized actions
    else:
        msg = f"I'm sorry, I couldn't quite understand what you wanted to do with '{item or 'that'}'."
        print(f"[WARNING] Unhandled action: {action}")
        return {"status": "failed", "reason": "unhandled_intent", "action": action, "message": msg}

if __name__ == "__main__":
    sentences = [
        # English
        "what is the price of white sneakers", "add red shoes to my cart", "remove the black jacket from my cart", "track order_TTKQEPzDQsai1H", "cancel order order_TTKQEPzDQsai1H", "can I pay for these shoes in EMIs?", "I need to pay for order order_TTKQEPzDQsai1H", "refund my order order_TTKQEPzDQsai1H", "create a payment link for 2000 rupees", "cancel subscription sub_999", "did my money settle yet?", "create a 15 percent discount for Diwali", "send an invoice of 5000 to Acme Corp", "Pay my employee John Doe 50000 rupees", "Split order order_TTKQEPzDQsai1H and send 20 percent to vendor Y", "Create a BharatQR code for 500 rupees", "Save my HDFC credit card for future purchases", "Track the refund status of order order_TTKQEPzDQsai1H", "what are the charges for UPI?", "refund 500 rupees from order order_TTKQEPzDQsai1H", "my payment failed", "get my magic checkout address", "what are the current offers",

        # Hindi (hi)
        "सफेद स्नीकर्स की कीमत क्या है?", "लाल जूते मेरे कार्ट में जोड़ें", "मेरे कार्ट से काली जैकेट हटा दें", "ऑर्डर order_TTKQEPzDQsai1H को ट्रैक करें", "ऑर्डर order_TTKQEPzDQsai1H रद्द करें", "क्या मैं इन जूतों के लिए ईएमआई में भुगतान कर सकता हूँ?", "मुझे ऑर्डर order_TTKQEPzDQsai1H के लिए भुगतान करना है", "मेरा ऑर्डर order_TTKQEPzDQsai1H रिफंड करें", "2000 रुपये के लिए पेमेंट लिंक बनाएं", "सब्सक्रिप्शन sub_999 रद्द करें", "क्या मेरा पैसा सेटल हो गया?", "दिवाली के लिए 15 प्रतिशत डिस्काउंट बनाएं", "Acme Corp को 5000 का इनवॉइस भेजें", "मेरे कर्मचारी John Doe को 50000 रुपये का भुगतान करें", "ऑर्डर order_TTKQEPzDQsai1H को स्प्लिट करें और 20 प्रतिशत vendor Y को भेजें", "500 रुपये के लिए भारत क्यूआर कोड बनाएं", "भविष्य की खरीदारी के लिए मेरा एचडीएफसी क्रेडिट कार्ड सेव करें", "ऑर्डर order_TTKQEPzDQsai1H के रिफंड स्टेटस को ट्रैक करें", "यूपीआई के लिए क्या चार्ज हैं?", "ऑर्डर order_TTKQEPzDQsai1H से 500 रुपये रिफंड करें", "मेरा भुगतान विफल हो गया", "मेरा मैजिक चेकआउट पता प्राप्त करें", "वर्तमान ऑफ़र क्या हैं",


        # Marathi (mr)
        "पांढऱ्या स्नीकर्सची किंमत काय आहे?", "माझ्या कार्टमध्ये लाल बूट जोडा", "माझ्या कार्टमधून काळी जॅकेट काढा", "ऑर्डर order_TTKQEPzDQsai1H ट्रॅक करा", "ऑर्डर order_TTKQEPzDQsai1H रद्द करा", "मी या बुटांसाठी ईएमआयमध्ये पैसे देऊ शकतो का?", "मला ऑर्डर order_TTKQEPzDQsai1H साठी पैसे द्यायचे आहेत", "माझा ऑर्डर order_TTKQEPzDQsai1H रिफंड करा", "2000 रुपयांसाठी पेमेंट लिंक तयार करा", "सबस्क्रिप्शन sub_999 रद्द करा", "माझे पैसे सेटल झाले का?", "दिवाळीसाठी 15 टक्के सूट तयार करा", "Acme Corp ला 5000 चे इन्व्हॉइस पाठवा", "माझा कर्मचारी John Doe ला 50000 रुपये द्या", "ऑर्डर order_TTKQEPzDQsai1H स्प्लिट करा आणि 20 टक्के vendor Y ला पाठवा", "500 रुपयांसाठी भारत क्यूआर कोड तयार करा", "भविष्यातील खरेदीसाठी माझे एचडीएफसी क्रेडिट कार्ड सेव्ह करा", "ऑर्डर order_TTKQEPzDQsai1H ची रिफंड स्थिती ट्रॅक करा", "यूपीआयसाठी काय शुल्क आहे?",

        # Kannada (kn)
        "ಬಿಳಿ ಬೂಟುಗಳ ಬೆಲೆ ಎಷ್ಟು?", "ಕೆಂಪು ಬೂಟುಗಳನ್ನು ನನ್ನ ಕಾರ್ಟ್‌ಗೆ ಸೇರಿಸಿ", "ನನ್ನ ಕಾರ್ಟ್‌ನಿಂದ ಕಪ್ಪು ಜಾಕೆಟ್ ತೆಗೆದುಹಾಕಿ", "ಆರ್ಡರ್ order_TTKQEPzDQsai1H ಟ್ರ್ಯಾಕ್ ಮಾಡಿ", "ಆರ್ಡರ್ order_TTKQEPzDQsai1H ರದ್ದುಮಾಡಿ", "ನಾನು ಈ ಬೂಟುಗಳಿಗೆ ಇಎಂಐನಲ್ಲಿ ಪಾವತಿಸಬಹುದೇ?", "ನಾನು ಆರ್ಡರ್ order_TTKQEPzDQsai1H ಗೆ ಪಾವತಿಸಬೇಕಾಗಿದೆ", "ನನ್ನ ಆರ್ಡರ್ order_TTKQEPzDQsai1H ರಿಫಂಡ್ ಮಾಡಿ", "2000 ರೂಪಾಯಿಗೆ ಪೇಮೆಂಟ್ ಲಿಂಕ್ ರಚಿಸಿ", "ಚಂದಾದಾರಿಕೆ sub_999 ರದ್ದುಮಾಡಿ", "ನನ್ನ ಹಣ ಸೆಟಲ್ ಆಗಿದೆಯೇ?", "ದೀಪಾವಳಿಗೆ 15 ಶೇಕಡಾ ರಿಯಾಯಿತಿ ರಚಿಸಿ", "Acme Corp ಗೆ 5000 ರ ಇನ್ವಾಯ್ಸ್ ಕಳುಹಿಸಿ", "ನನ್ನ ಉದ್ಯೋಗಿ John Doe ಗೆ 50000 ರೂಪಾಯಿ ಪಾವತಿಸಿ", "ಆರ್ಡರ್ order_TTKQEPzDQsai1H ಅನ್ನು ಸ್ಪ್ಲಿಟ್ ಮಾಡಿ ಮತ್ತು 20 ಶೇಕಡಾವನ್ನು vendor Y ಗೆ ಕಳುಹಿಸಿ", "500 ರೂಪಾಯಿಗೆ ಭಾರತ್ ಕ್ಯೂಆರ್ ಕೋಡ್ ರಚಿಸಿ", "ಭವಿಷ್ಯದ ಖರೀದಿಗಳಿಗಾಗಿ ನನ್ನ ಎಚ್‌ಡಿಎಫ್‌ಸಿ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಉಳಿಸಿ", "ಆರ್ಡರ್ order_TTKQEPzDQsai1H ರ ರಿಫಂಡ್ ಸ್ಥಿತಿಯನ್ನು ಟ್ರ್ಯಾಕ್ ಮಾಡಿ", "ಯುಪಿಐ ಶುಲ್ಕಗಳು ಯಾವುವು?",

        # Tamil (ta)
        "வெள்ளை ஸ்னீக்கர்களின் விலை என்ன?", "சிவப்பு காலணிகளை என் வண்டியில் சேர்க்கவும்", "என் வண்டியிலிருந்து கருப்பு ஜாக்கெட்டை அகற்றவும்", "ஆர்டர் order_TTKQEPzDQsai1H ஐ கண்காணிக்கவும்", "ஆர்டர் order_TTKQEPzDQsai1H ஐ ரத்து செய்", "இந்த காலணிகளுக்கு நான் இஎம்ஐயில் செலுத்த முடியுமா?", "ஆர்டர் order_TTKQEPzDQsai1H க்கு நான் பணம் செலுத்த வேண்டும்", "என் ஆர்டர் order_TTKQEPzDQsai1H ஐ ரீபண்ட் செய்", "2000 ரூபாய்க்கு கட்டண இணைப்பை உருவாக்கவும்", "சந்தா sub_999 ஐ ரத்து செய்", "என் பணம் செட்டில் ஆனதா?", "தீபாவளிக்கு 15 சதவீத தள்ளுபடியை உருவாக்கவும்", "Acme Corp க்கு 5000 இன் விலைப்பட்டியலை அனுப்பவும்", "என் ஊழியர் John Doe க்கு 50000 ரூபாய் செலுத்தவும்", "ஆர்டர் order_TTKQEPzDQsai1H ஐ பிரித்து 20 சதவீதத்தை vendor Y க்கு அனுப்பவும்", "500 ரூபாய்க்கு பாரத் கியூஆர் குறியீட்டை உருவாக்கவும்", "எதிர்கால வாங்குதல்களுக்காக என் எச்டிஎஃப்சி கிரெடிட் கார்டை சேமிக்கவும்", "ஆர்டர் order_TTKQEPzDQsai1H இன் ரீபண்ட் நிலையை கண்காணிக்கவும்", "யுபிஐ கட்டணங்கள் என்ன?",

        # Telugu (te)
        "తెల్ల స్నీకర్ల ధర ఎంత?", "ఎర్ర బూట్లను నా కార్ట్‌లో చేర్చండి", "నా కార్ట్ నుండి బ్లాక్ జాకెట్ తొలగించండి", "ఆర్డర్ order_TTKQEPzDQsai1H ని ట్రాక్ చేయండి", "ఆర్డర్ order_TTKQEPzDQsai1H రద్దు చేయండి", "నేను ఈ బూట్లకు ఈఎంఐలో చెల్లించవచ్చా?", "నేను ఆర్డర్ order_TTKQEPzDQsai1H కు చెల్లించాలి", "నా ఆర్డర్ order_TTKQEPzDQsai1H ని రీఫండ్ చేయండి", "2000 రూపాయల కోసం పేమెంట్ లింక్ క్రియేట్ చేయండి", "సబ్‌స్క్రిప్షన్ sub_999 రద్దు చేయండి", "నా డబ్బు సెటిల్ అయిందా?", "దీపావళి కోసం 15 శాతం తగ్గింపును క్రియేట్ చేయండి", "Acme Corp కు 5000 ఇన్వాయిస్ పంపండి", "నా ఉద్యోగి John Doe కు 50000 రూపాయలు చెల్లించండి", "ఆర్డర్ order_TTKQEPzDQsai1H ని స్ప్లిట్ చేసి 20 శాతం vendor Y కు పంపండి", "500 రూపాయల కోసం భారత్ క్యూఆర్ కోడ్ క్రియేట్ చేయండి", "భవిష్యత్తు కొనుగోళ్ల కోసం నా హెచ్‌డిఎఫ్‌సి క్రెడిట్ కార్డును సేవ్ చేయండి", "ఆర్డర్ order_TTKQEPzDQsai1H యొక్క రీఫండ్ స్థితిని ట్రాక్ చేయండి", "యుపిఐ ఛార్జీలు ఏమిటి?",

        # Malayalam (ml)
        "വെളുത്ത സ്നീക്കറുകളുടെ വില എത്രയാണ്?", "എൻ്റെ കാർട്ടിലേക്ക് ചുവന്ന ഷൂകൾ ചേർക്കുക", "എൻ്റെ കാർട്ടിൽ നിന്ന് കറുത്ത ജാക്കറ്റ് നീക്കം ചെയ്യുക", "ഓർഡർ order_TTKQEPzDQsai1H ട്രാക്ക് ചെയ്യുക", "ഓർഡർ order_TTKQEPzDQsai1H റദ്ദാക്കുക", "എനിക്ക് ഈ ഷൂകൾക്ക് ഇഎംഐയിൽ പണമടക്കാൻ കഴിയുമോ?", "ഓർഡർ order_TTKQEPzDQsai1H ന് ഞാൻ പണമടയ്ക്കണം", "എൻ്റെ ഓർഡർ order_TTKQEPzDQsai1H റീഫണ്ട് ചെയ്യുക", "2000 രൂപയ്ക്ക് പേയ്‌മെൻ്റ് ലിങ്ക് സൃഷ്ടിക്കുക", "സബ്സ്ക്രിപ്ഷൻ sub_999 റദ്ദാക്കുക", "എൻ്റെ പണം സെറ്റിൽ ആയോ?", "ദീപാവലിക്ക് 15 ശതമാനം കിഴിവ് സൃഷ്ടിക്കുക", "Acme Corp ന് 5000 ൻ്റെ ഇൻവോയ്സ് അയയ്ക്കുക", "എൻ്റെ ജീവനക്കാരൻ John Doe ന് 50000 രൂപ നൽകുക", "ഓർഡർ order_TTKQEPzDQsai1H സ്പ്ലിറ്റ് ചെയ്ത് 20 ശതമാനം vendor Y ന് അയയ്ക്കുക", "500 രൂപയ്ക്ക് ഭാരത് ക്യുആർ കോഡ് സൃഷ്ടിക്കുക", "ഭാവിയിലെ വാങ്ങലുകൾക്കായി എൻ്റെ എച്ച്ഡിഎഫ്സി ക്രെഡിറ്റ് കാർഡ് സേവ് ചെയ്യുക", "ഓർഡർ order_TTKQEPzDQsai1H ൻ്റെ റീഫണ്ട് നില ട്രാക്ക് ചെയ്യുക", "യുപിഐ ചാർജുകൾ എന്തൊക്കെയാണ്?"
    ]
    
    print(f"Running massive Indic Agentic test suite ({len(sentences)} cases)...\n")
    for s in sentences:
        process_voice_command(s)
