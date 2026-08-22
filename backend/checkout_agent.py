import json
import os
import sys
import difflib
sys.stdout.reconfigure(encoding="utf-8")
from ai_pipeline import intent_pipeline

def load_inventory():
    path = os.path.join(os.path.dirname(__file__), "inventory.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Mock databases for the remaining intents
MOCK_CART = {"black jacket": {"qty": 1}}
MOCK_ORDERS = {
    "123": {"status": "shipped", "total": 2500},
    "456": {"status": "processing", "total": 1800}
}
MOCK_SETTLEMENTS = {
    "setl_789": {"status": "settled", "amount": 50000, "date": "2023-10-25"}
}
MOCK_OFFERS = {}
MOCK_INVOICES = {}
MOCK_PAYOUTS = {}
MOCK_SAVED_CARDS = []
MOCK_REFUNDS = {"456": {"status": "processed"}}
MOCK_SUBSCRIPTIONS = {
    "sub_444": {"status": "active", "plan": "Monthly Premium"}
}

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

def process_voice_command(query: str):
    inventory = load_inventory()
    
    # 1. AI Extract Intent
    print(f"\nUser Said: '{query}'")
    intent = intent_pipeline.extract_intent(query)
    
    # 1.5 Handle Gated Money Actions (Track 1 Bar) - MUST BE AT TOP
    if intent.get('requires_confirmation'):
        msg = "This action requires money movement. Initiating OTP Gate... Please confirm before we trigger the Razorpay API."
        print(f"[AUDIT] {msg}")
        return {"status": "pending_confirmation", "message": msg}
    
    action = intent.get('action')
    item = intent.get('item')
    requested_qty = intent.get('quantity', 1)
    
    # 2. Handle Price Checking (Search)
    if action in ['search', 'price_check', 'find', 'query'] and item:
        matches = difflib.get_close_matches(item, inventory.keys(), n=1, cutoff=0.6)
        if matches:
            item = matches[0]
            price = inventory[item]['price']
            msg = f"The {item} costs ₹{price}. Would you like me to add it to your cart?"
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "price_check", "message": msg}
        else:
            alt = suggest_any_alternative(inventory, item)
            if alt:
                msg = f"I couldn't find '{item}', but we do have {alt}. Would you like to check its price instead?"
            else:
                msg = f"I couldn't find '{item}' in our catalog."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "item_not_found", "alternative": alt, "message": msg}

    # 3. Handle Add/Checkout Actions
    elif action in ['add', 'checkout', 'buy', 'purchase', 'add_to_cart'] and item:
        matches = difflib.get_close_matches(item, inventory.keys(), n=1, cutoff=0.6)
        if matches:
            item = matches[0]
            stock = inventory[item]['stock']
            
            # --- THE GRACEFUL FAILURE (OUT OF STOCK) ---
            if stock == 0:
                alt = suggest_alternative(inventory, item)
                msg = f"Sorry, {item} is currently out of stock."
                if alt:
                    msg += f" However, we do have {alt} in stock! Should I add that instead?"
                print(f"[AGENT] {msg}")
                return {"status": "failed", "reason": "out_of_stock", "alternative": alt, "message": msg}
                
            # --- PARTIAL FULFILLMENT (QUANTITY MISMATCH) ---
            elif stock < requested_qty:
                msg = f"We only have {stock} {item}(s) in stock, but you asked for {requested_qty}. Would you like me to add all {stock} to your cart instead?"
                print(f"[AGENT] {msg}")
                return {"status": "pending_quantity_confirmation", "available": stock, "message": msg}
                
            # --- THE HAPPY PATH + UPSELL ---
            msg = f"Successfully added {requested_qty} {item}(s) to your cart."
            if intent.get('cross_sell'):
                msg += f" Would you also like to add {intent['cross_sell']} to your order?"
            print(f"[AGENT] {msg}")
                
            return {"status": "success", "action": "added", "message": msg}
            
        else:
            alt = suggest_any_alternative(inventory, item)
            if alt:
                msg = f"I couldn't find '{item}'. But we do have {alt} in stock! Would you like me to add that instead?"
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
        
    # 5. Handle Remove from Cart
    elif action in ['remove', 'delete', 'remove_from_cart', 'remove_item'] and item:
        matches = difflib.get_close_matches(item, MOCK_CART.keys(), n=1, cutoff=0.6)
        if matches:
            msg = f"Successfully removed {matches[0]} from your cart."
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "removed", "message": msg}
        else:
            msg = f"You don't have '{item}' in your cart to remove."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "not_in_cart", "message": msg}
            
    # 6. Handle Order Tracking
    elif action in ['track', 'check_status', 'order_status']:
        order_id = intent.get('order_id')
        if order_id in MOCK_ORDERS:
            status = MOCK_ORDERS[order_id]['status']
            msg = f"Order {order_id} is currently: {status.upper()}."
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "track", "order_status": status, "message": msg}
        else:
            msg = f"I couldn't find order {order_id}."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "order_not_found", "message": msg}

    # 7. Handle E-Commerce Order Cancelation (Non-subscription)
    elif action in ['cancel', 'cancel_order']:
        order_id = intent.get('order_id')
        if order_id in MOCK_ORDERS:
            if MOCK_ORDERS[order_id]['status'] == 'shipped':
                msg = f"Sorry, order {order_id} has already shipped and cannot be canceled."
                print(f"[AGENT] {msg}")
                return {"status": "failed", "reason": "already_shipped", "message": msg}
            else:
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
        msg = f"Successfully initiated refund for order {order_id}."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "refund", "message": msg}

    # 7.3 Handle Payment Links (Merchant side)
    elif action in ['create_payment_link', 'payment_link', 'generate_payment_link']:
        amount = intent.get('amount')
        msg = f"Generated a Razorpay Payment Link for ₹{amount}."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "create_payment_link", "message": msg}

    # 7.5 Handle Subscription Cancellation (Customer side)
    elif action in ['cancel_subscription', 'stop_subscription']:
        sub_id = intent.get('subscription_id')
        if sub_id in MOCK_SUBSCRIPTIONS:
            if MOCK_SUBSCRIPTIONS[sub_id]['status'] == 'canceled':
                msg = f"Subscription {sub_id} is already canceled."
                print(f"[AGENT] {msg}")
                return {"status": "failed", "reason": "already_canceled", "message": msg}
            else:
                MOCK_SUBSCRIPTIONS[sub_id]['status'] = 'canceled'
                msg = f"Your subscription {sub_id} has been successfully canceled. No further charges will be made."
                print(f"[AGENT] {msg}")
                return {"status": "success", "action": "cancel_subscription", "message": msg}
        else:
            msg = f"I couldn't find an active subscription with ID {sub_id}."
            print(f"[WARNING] {msg}")
            return {"status": "failed", "reason": "subscription_not_found", "message": msg}

    # 8. Handle Settlement Check (Merchant side)
    elif action in ['check_settlement', 'settlement_status', 'check_settlements']:
        msg = "Checking your latest settlements... Your last settlement setl_789 for ₹50000 was settled on 2023-10-25."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "settlement_checked", "message": msg}

    # 9. Handle Smart Offers (Merchant side)
    elif action in ['create_offer', 'create_discount', 'offer', 'discount']:
        discount = intent.get('discount_percent', 10)
        offer_id = f"offer_diwali_{discount}"
        MOCK_OFFERS[offer_id] = {"discount": discount, "status": "active"}
        msg = f"Successfully created a {discount}% Razorpay Offer: {offer_id}"
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "create_offer", "offer_id": offer_id, "message": msg}

    # 10. Handle B2B Invoicing (Merchant side)
    elif action in ['create_invoice', 'invoice', 'generate_invoice']:
        amount = intent.get('amount', 0.0)
        company = intent.get('company', 'Customer')
        inv_id = f"inv_{len(MOCK_INVOICES) + 1000}"
        MOCK_INVOICES[inv_id] = {"amount": amount, "company": company, "status": "issued"}
        msg = f"Generated GST Invoice {inv_id} for {company} for ₹{amount}."
        print(f"[AGENT] {msg}")
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
        payout_id = f"pout_{len(MOCK_PAYOUTS) + 1000}"
        MOCK_PAYOUTS[payout_id] = {"amount": amount, "recipient": recipient, "status": "processed"}
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
        MOCK_SAVED_CARDS.append(network)
        msg = f"Your {network} card has been securely tokenized and saved via Razorpay TokenHQ for future purchases."
        print(f"[AGENT] {msg}")
        return {"status": "success", "action": "save_card", "message": msg}

    # 16. Handle Refund Tracking (Customer side)
    elif action == 'track_refund':
        order_id = intent.get('order_id')
        if order_id in MOCK_REFUNDS:
            status = MOCK_REFUNDS[order_id]['status']
            msg = f"Your refund for order {order_id} has been {status} by the bank. Bank ARN: 8493820498."
            print(f"[AGENT] {msg}")
            return {"status": "success", "action": "track_refund", "message": msg}
        else:
            msg = f"I couldn't find a refund record for order {order_id}."
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
        "what is the price of white sneakers", "add red shoes to my cart", "remove the black jacket from my cart", "track order 123", "cancel order 456", "can I pay for these shoes in EMIs?", "I need to pay for order 12345", "refund my order 98765", "create a payment link for 2000 rupees", "cancel subscription sub_999", "did my money settle yet?", "create a 15 percent discount for Diwali", "send an invoice of 5000 to Acme Corp", "Pay my employee John Doe 50000 rupees", "Split order 123 and send 20 percent to vendor Y", "Create a BharatQR code for 500 rupees", "Save my HDFC credit card for future purchases", "Track the refund status of order 456", "what are the charges for UPI?",

        # Hindi (hi)
        "सफेद स्नीकर्स की कीमत क्या है?", "लाल जूते मेरे कार्ट में जोड़ें", "मेरे कार्ट से काली जैकेट हटा दें", "ऑर्डर 123 को ट्रैक करें", "ऑर्डर 456 रद्द करें", "क्या मैं इन जूतों के लिए ईएमआई में भुगतान कर सकता हूँ?", "मुझे ऑर्डर 12345 के लिए भुगतान करना है", "मेरा ऑर्डर 98765 रिफंड करें", "2000 रुपये के लिए पेमेंट लिंक बनाएं", "सब्सक्रिप्शन sub_999 रद्द करें", "क्या मेरा पैसा सेटल हो गया?", "दिवाली के लिए 15 प्रतिशत डिस्काउंट बनाएं", "Acme Corp को 5000 का इनवॉइस भेजें", "मेरे कर्मचारी John Doe को 50000 रुपये का भुगतान करें", "ऑर्डर 123 को स्प्लिट करें और 20 प्रतिशत vendor Y को भेजें", "500 रुपये के लिए भारत क्यूआर कोड बनाएं", "भविष्य की खरीदारी के लिए मेरा एचडीएफसी क्रेडिट कार्ड सेव करें", "ऑर्डर 456 के रिफंड स्टेटस को ट्रैक करें", "यूपीआई के लिए क्या चार्ज हैं?",

        # Marathi (mr)
        "पांढऱ्या स्नीकर्सची किंमत काय आहे?", "माझ्या कार्टमध्ये लाल बूट जोडा", "माझ्या कार्टमधून काळी जॅकेट काढा", "ऑर्डर 123 ट्रॅक करा", "ऑर्डर 456 रद्द करा", "मी या बुटांसाठी ईएमआयमध्ये पैसे देऊ शकतो का?", "मला ऑर्डर 12345 साठी पैसे द्यायचे आहेत", "माझा ऑर्डर 98765 रिफंड करा", "2000 रुपयांसाठी पेमेंट लिंक तयार करा", "सबस्क्रिप्शन sub_999 रद्द करा", "माझे पैसे सेटल झाले का?", "दिवाळीसाठी 15 टक्के सूट तयार करा", "Acme Corp ला 5000 चे इन्व्हॉइस पाठवा", "माझा कर्मचारी John Doe ला 50000 रुपये द्या", "ऑर्डर 123 स्प्लिट करा आणि 20 टक्के vendor Y ला पाठवा", "500 रुपयांसाठी भारत क्यूआर कोड तयार करा", "भविष्यातील खरेदीसाठी माझे एचडीएफसी क्रेडिट कार्ड सेव्ह करा", "ऑर्डर 456 ची रिफंड स्थिती ट्रॅक करा", "यूपीआयसाठी काय शुल्क आहे?",

        # Kannada (kn)
        "ಬಿಳಿ ಬೂಟುಗಳ ಬೆಲೆ ಎಷ್ಟು?", "ಕೆಂಪು ಬೂಟುಗಳನ್ನು ನನ್ನ ಕಾರ್ಟ್‌ಗೆ ಸೇರಿಸಿ", "ನನ್ನ ಕಾರ್ಟ್‌ನಿಂದ ಕಪ್ಪು ಜಾಕೆಟ್ ತೆಗೆದುಹಾಕಿ", "ಆರ್ಡರ್ 123 ಟ್ರ್ಯಾಕ್ ಮಾಡಿ", "ಆರ್ಡರ್ 456 ರದ್ದುಮಾಡಿ", "ನಾನು ಈ ಬೂಟುಗಳಿಗೆ ಇಎಂಐನಲ್ಲಿ ಪಾವತಿಸಬಹುದೇ?", "ನಾನು ಆರ್ಡರ್ 12345 ಗೆ ಪಾವತಿಸಬೇಕಾಗಿದೆ", "ನನ್ನ ಆರ್ಡರ್ 98765 ರಿಫಂಡ್ ಮಾಡಿ", "2000 ರೂಪಾಯಿಗೆ ಪೇಮೆಂಟ್ ಲಿಂಕ್ ರಚಿಸಿ", "ಚಂದಾದಾರಿಕೆ sub_999 ರದ್ದುಮಾಡಿ", "ನನ್ನ ಹಣ ಸೆಟಲ್ ಆಗಿದೆಯೇ?", "ದೀಪಾವಳಿಗೆ 15 ಶೇಕಡಾ ರಿಯಾಯಿತಿ ರಚಿಸಿ", "Acme Corp ಗೆ 5000 ರ ಇನ್ವಾಯ್ಸ್ ಕಳುಹಿಸಿ", "ನನ್ನ ಉದ್ಯೋಗಿ John Doe ಗೆ 50000 ರೂಪಾಯಿ ಪಾವತಿಸಿ", "ಆರ್ಡರ್ 123 ಅನ್ನು ಸ್ಪ್ಲಿಟ್ ಮಾಡಿ ಮತ್ತು 20 ಶೇಕಡಾವನ್ನು vendor Y ಗೆ ಕಳುಹಿಸಿ", "500 ರೂಪಾಯಿಗೆ ಭಾರತ್ ಕ್ಯೂಆರ್ ಕೋಡ್ ರಚಿಸಿ", "ಭವಿಷ್ಯದ ಖರೀದಿಗಳಿಗಾಗಿ ನನ್ನ ಎಚ್‌ಡಿಎಫ್‌ಸಿ ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್ ಉಳಿಸಿ", "ಆರ್ಡರ್ 456 ರ ರಿಫಂಡ್ ಸ್ಥಿತಿಯನ್ನು ಟ್ರ್ಯಾಕ್ ಮಾಡಿ", "ಯುಪಿಐ ಶುಲ್ಕಗಳು ಯಾವುವು?",

        # Tamil (ta)
        "வெள்ளை ஸ்னீக்கர்களின் விலை என்ன?", "சிவப்பு காலணிகளை என் வண்டியில் சேர்க்கவும்", "என் வண்டியிலிருந்து கருப்பு ஜாக்கெட்டை அகற்றவும்", "ஆர்டர் 123 ஐ கண்காணிக்கவும்", "ஆர்டர் 456 ஐ ரத்து செய்", "இந்த காலணிகளுக்கு நான் இஎம்ஐயில் செலுத்த முடியுமா?", "ஆர்டர் 12345 க்கு நான் பணம் செலுத்த வேண்டும்", "என் ஆர்டர் 98765 ஐ ரீபண்ட் செய்", "2000 ரூபாய்க்கு கட்டண இணைப்பை உருவாக்கவும்", "சந்தா sub_999 ஐ ரத்து செய்", "என் பணம் செட்டில் ஆனதா?", "தீபாவளிக்கு 15 சதவீத தள்ளுபடியை உருவாக்கவும்", "Acme Corp க்கு 5000 இன் விலைப்பட்டியலை அனுப்பவும்", "என் ஊழியர் John Doe க்கு 50000 ரூபாய் செலுத்தவும்", "ஆர்டர் 123 ஐ பிரித்து 20 சதவீதத்தை vendor Y க்கு அனுப்பவும்", "500 ரூபாய்க்கு பாரத் கியூஆர் குறியீட்டை உருவாக்கவும்", "எதிர்கால வாங்குதல்களுக்காக என் எச்டிஎஃப்சி கிரெடிட் கார்டை சேமிக்கவும்", "ஆர்டர் 456 இன் ரீபண்ட் நிலையை கண்காணிக்கவும்", "யுபிஐ கட்டணங்கள் என்ன?",

        # Telugu (te)
        "తెల్ల స్నీకర్ల ధర ఎంత?", "ఎర్ర బూట్లను నా కార్ట్‌లో చేర్చండి", "నా కార్ట్ నుండి బ్లాక్ జాకెట్ తొలగించండి", "ఆర్డర్ 123 ని ట్రాక్ చేయండి", "ఆర్డర్ 456 రద్దు చేయండి", "నేను ఈ బూట్లకు ఈఎంఐలో చెల్లించవచ్చా?", "నేను ఆర్డర్ 12345 కు చెల్లించాలి", "నా ఆర్డర్ 98765 ని రీఫండ్ చేయండి", "2000 రూపాయల కోసం పేమెంట్ లింక్ క్రియేట్ చేయండి", "సబ్‌స్క్రిప్షన్ sub_999 రద్దు చేయండి", "నా డబ్బు సెటిల్ అయిందా?", "దీపావళి కోసం 15 శాతం తగ్గింపును క్రియేట్ చేయండి", "Acme Corp కు 5000 ఇన్వాయిస్ పంపండి", "నా ఉద్యోగి John Doe కు 50000 రూపాయలు చెల్లించండి", "ఆర్డర్ 123 ని స్ప్లిట్ చేసి 20 శాతం vendor Y కు పంపండి", "500 రూపాయల కోసం భారత్ క్యూఆర్ కోడ్ క్రియేట్ చేయండి", "భవిష్యత్తు కొనుగోళ్ల కోసం నా హెచ్‌డిఎఫ్‌సి క్రెడిట్ కార్డును సేవ్ చేయండి", "ఆర్డర్ 456 యొక్క రీఫండ్ స్థితిని ట్రాక్ చేయండి", "యుపిఐ ఛార్జీలు ఏమిటి?",

        # Malayalam (ml)
        "വെളുത്ത സ്നീക്കറുകളുടെ വില എത്രയാണ്?", "എൻ്റെ കാർട്ടിലേക്ക് ചുവന്ന ഷൂകൾ ചേർക്കുക", "എൻ്റെ കാർട്ടിൽ നിന്ന് കറുത്ത ജാക്കറ്റ് നീക്കം ചെയ്യുക", "ഓർഡർ 123 ട്രാക്ക് ചെയ്യുക", "ഓർഡർ 456 റദ്ദാക്കുക", "എനിക്ക് ഈ ഷൂകൾക്ക് ഇഎംഐയിൽ പണമടക്കാൻ കഴിയുമോ?", "ഓർഡർ 12345 ന് ഞാൻ പണമടയ്ക്കണം", "എൻ്റെ ഓർഡർ 98765 റീഫണ്ട് ചെയ്യുക", "2000 രൂപയ്ക്ക് പേയ്‌മെൻ്റ് ലിങ്ക് സൃഷ്ടിക്കുക", "സബ്സ്ക്രിപ്ഷൻ sub_999 റദ്ദാക്കുക", "എൻ്റെ പണം സെറ്റിൽ ആയോ?", "ദീപാവലിക്ക് 15 ശതമാനം കിഴിവ് സൃഷ്ടിക്കുക", "Acme Corp ന് 5000 ൻ്റെ ഇൻവോയ്സ് അയയ്ക്കുക", "എൻ്റെ ജീവനക്കാരൻ John Doe ന് 50000 രൂപ നൽകുക", "ഓർഡർ 123 സ്പ്ലിറ്റ് ചെയ്ത് 20 ശതമാനം vendor Y ന് അയയ്ക്കുക", "500 രൂപയ്ക്ക് ഭാരത് ക്യുആർ കോഡ് സൃഷ്ടിക്കുക", "ഭാവിയിലെ വാങ്ങലുകൾക്കായി എൻ്റെ എച്ച്ഡിഎഫ്സി ക്രെഡിറ്റ് കാർഡ് സേവ് ചെയ്യുക", "ഓർഡർ 456 ൻ്റെ റീഫണ്ട് നില ട്രാക്ക് ചെയ്യുക", "യുപിഐ ചാർജുകൾ എന്തൊക്കെയാണ്?"
    ]
    
    print(f"Running massive Indic Agentic test suite ({len(sentences)} cases)...\n")
    for s in sentences:
        process_voice_command(s)
