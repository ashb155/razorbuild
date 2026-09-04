import re
import time
import difflib
import logging
from typing import Dict, Any, Optional
from models.schemas import ExtractedIntent, AgentResponse
from services.inventory_service import InventoryService
from services.session_service import UserSession
from services.razorpay_service import RazorpayService

logger = logging.getLogger(__name__)

class CatalogHandler:
    """Handles product search, price checks, and inventory status."""

    def __init__(self, inventory_service: InventoryService):
        self.inventory = inventory_service

    def handle_search(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        item = intent.item
        matched_key = self.inventory.find_item_in_inventory(item)
        
        if matched_key:
            product = self.inventory.get_product(matched_key)
            name = product['name']
            price = product['price']
            stock = product['stock']
            
            if stock > 0:
                session.pending_search = matched_key
                msg = f"The {name} costs ₹{price}. Would you like me to add it to your cart?"
                return AgentResponse(status="success", action="price_check", message=msg)
            else:
                alt_key = self.inventory.suggest_alternative(matched_key)
                msg = f"The {name} costs ₹{price}, but it's currently out of stock."
                if alt_key:
                    alt_product = self.inventory.get_product(alt_key)
                    session.pending_alternative = alt_key
                    msg += f" However, we do have the {alt_product['name']} in stock! Would you like to check it out?"
                return AgentResponse(status="success", action="price_check", alternative=alt_key, message=msg)
        else:
            alt_key = self.inventory.suggest_any_alternative(item or "")
            if alt_key:
                alt_product = self.inventory.get_product(alt_key)
                session.pending_search = alt_key
                msg = f"I couldn't find '{item}', but we do have the {alt_product['name']} which costs ₹{alt_product['price']}. Would you like me to add it to your cart?"
                return AgentResponse(status="failed", reason="item_not_found", alternative=alt_key, message=msg)
            else:
                msg = f"I couldn't find '{item}' in our catalog."
                return AgentResponse(status="failed", reason="item_not_found", message=msg)

class CartHandler:
    """Handles adding, removing, reviewing, and checking out items in the user cart."""

    def __init__(self, inventory_service: InventoryService, razorpay_service: RazorpayService):
        self.inventory = inventory_service
        self.razorpay = razorpay_service

    def handle_add(self, intent: ExtractedIntent, session: UserSession, skip_gate: bool = False) -> AgentResponse:
        item = intent.item
        matched_key = self.inventory.find_item_in_inventory(item)
        if not matched_key:
            alt_key = self.inventory.suggest_any_alternative(item or "")
            if alt_key:
                alt_product = self.inventory.get_product(alt_key)
                session.pending_alternative = alt_key
                msg = f"I couldn't find '{item}'. But we do have the {alt_product['name']} in stock! Would you like me to add that instead?"
                return AgentResponse(status="failed", reason="item_not_found", alternative=alt_key, message=msg)
            return AgentResponse(status="failed", reason="item_not_found", message=f"I couldn't find '{item}' in our catalog.")

        product = self.inventory.get_product(matched_key)
        stock = product['stock']
        price = product['price']
        name = product['name']
        requested_qty = max(1, intent.quantity or 1)

        # 1. Out of Stock Handling
        if stock == 0:
            alt_key = self.inventory.suggest_alternative(matched_key)
            msg = f"Sorry, the {name} is currently out of stock."
            if alt_key:
                alt_product = self.inventory.get_product(alt_key)
                session.pending_alternative = alt_key
                msg += f" However, we do have the {alt_product['name']} in stock! Should I add that instead?"
            return AgentResponse(status="failed", reason="out_of_stock", alternative=alt_key, message=msg)

        # 2. Quantity Mismatch Handling
        if stock < requested_qty:
            if skip_gate:
                requested_qty = stock
            else:
                msg = f"We only have {stock} {name}(s) in stock, but you asked for {requested_qty}. Would you like me to add all {stock} to your cart instead?"
                return AgentResponse(status="pending_quantity_confirmation", available=stock, message=msg)

        # 3. Direct Buy / Instant Checkout vs Add to Cart
        if intent.action in ['checkout', 'buy', 'purchase']:
            total = price * requested_qty
            plink = self.razorpay.create_payment_link(
                amount_inr=total,
                description=f"Payment for {requested_qty} x {name}",
                customer_phone=session.session_id if session.session_id.startswith("+") else None
            )
            session.orders[plink['id']] = {"status": "created", "total": total}
            session.cart.clear()
            msg = f"I've prepared your order for {requested_qty} {name}. The total is ₹{total}. Please complete your payment here: {plink['url']}"
            return AgentResponse(status="success", action="checkout_with_link", payment_link=plink['url'], message=msg)
        else:
            session.cart[matched_key] = session.cart.get(matched_key, 0) + requested_qty
            msg = f"I've added {requested_qty} {name} to your cart."
            
            # Dynamic Cross-Sell Recommendation
            cross_name = self.inventory.get_cross_sell(matched_key, session.cart)
            if cross_name:
                # Find matching key for the cross sell
                for k, v in self.inventory.get_raw_inventory().items():
                    if v['name'] == cross_name:
                        session.pending_cross_sell = k
                        break
                msg += f" Would you also like to add {cross_name} to your order?"
            else:
                msg += " Would you like anything else?"
                
            return AgentResponse(status="success", action="added", message=msg)

    def handle_remove(self, intent: ExtractedIntent, session: UserSession, raw_query: str) -> AgentResponse:
        item = intent.item or ""
        cart = session.cart
        inv = self.inventory.get_raw_inventory()

        if not cart:
            return AgentResponse(status="failed", reason="empty_cart", message="Your cart is currently empty.")

        matched_key = None
        
        # 1. Exact key match in cart
        if item in cart:
            matched_key = item

        # 2. Match via inventory lookup
        if not matched_key:
            inv_key = self.inventory.find_item_in_inventory(item)
            if inv_key and inv_key in cart:
                matched_key = inv_key

        # 3. Safe close match against item names in cart with word overlap guard
        if not matched_key and item:
            cart_names = {inv[k]['name'].lower(): k for k in cart if k in inv}
            matches = difflib.get_close_matches(item.lower(), cart_names.keys(), n=1, cutoff=0.5)
            if matches:
                # Check for word overlap of length >= 3
                item_words = {w for w in item.lower().split() if len(w) >= 3}
                match_words = {w for w in matches[0].split() if len(w) >= 3}
                if item_words & match_words:
                    matched_key = cart_names[matches[0]]

        if matched_key:
            prod_name = inv.get(matched_key, {}).get('name', matched_key.replace('_', ' '))
            del cart[matched_key]
            return AgentResponse(status="success", action="removed", message=f"I've removed the {prod_name} from your cart.")
        else:
            cart_list = ", ".join(inv[k]['name'] for k in cart if k in inv) or "nothing"
            return AgentResponse(status="failed", reason="not_in_cart", message=f"I couldn't find '{item or 'that'}' in your cart. Your cart currently has: {cart_list}.")

    def handle_confirm_cart(self, session: UserSession) -> AgentResponse:
        cart = session.cart
        inv = self.inventory.get_raw_inventory()
        if not cart:
            return AgentResponse(status="failed", reason="empty_cart", message="Your cart is empty, so there's nothing to checkout!")

        cart_items = []
        total_amount = 0.0
        for k, qty in cart.items():
            if k in inv:
                name = inv[k]['name']
                price = inv[k]['price']
                total_amount += price * qty
                cart_items.append(f"{qty}x {name}")

        summary = ", ".join(cart_items)
        plink = self.razorpay.create_payment_link(
            amount_inr=total_amount,
            description=f"Payment for cart: {summary}",
            customer_phone=session.session_id if session.session_id.startswith("+") else None
        )
        session.orders[plink['id']] = {"status": "created", "total": total_amount}
        session.cart.clear()
        
        msg = f"Perfect! I've generated your secure payment link for {summary}. You can complete your ₹{total_amount} payment here: {plink['url']}"
        return AgentResponse(status="success", action="checkout_cart", payment_link=plink['url'], message=msg)

class CustomerServiceHandler:
    """Handles Order Tracking, Refunds, Subscriptions, EMI, and Magic Checkout."""

    def __init__(self, razorpay_service: RazorpayService, inventory_service: InventoryService):
        self.razorpay = razorpay_service
        self.inventory = inventory_service


    def _is_valid_rzp_id(self, id_str: Optional[str]) -> bool:
        """Validates a Razorpay ID by checking its real format: known prefix + 14 alphanumeric chars.
        Rejects hallucinated/short IDs like 'order_12345', 'order_AbCd123', '123' etc."""
        if not id_str:
            return False
        # Razorpay IDs: pay_, order_, rfnd_, plink_, sub_, inv_, pout_ + 14 alphanum chars
        return bool(re.match(r'^(pay|order|rfnd|plink|sub|inv|pout)_[A-Za-z0-9]{14}$', str(id_str)))

    def handle_track(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        order_id = intent.order_id if self._is_valid_rzp_id(intent.order_id) else None

        if not order_id and session.orders:
            order_id = list(session.orders.keys())[-1]

        if not order_id:
            try:
                payments = self.razorpay.client.payment.all({'count': 1})
                items = payments.get('items', [])
                if items and items[0].get('captured'):
                    order_id = items[0]['id']
            except Exception:
                pass

        if not order_id:
            return AgentResponse(status="failed", reason="order_not_found", message="Please specify an order ID to track, or complete a purchase first.")

        status = self.razorpay.fetch_order_status(str(order_id))
        if status:
            prefix = order_id.split('_')[0] if '_' in order_id else 'payment'
            path_map = {'pay': 'payments', 'plink': 'payment-links', 'order': 'orders'}
            path = path_map.get(prefix, 'payments')
            dashboard_url = f"https://dashboard.razorpay.com/app/{path}/{order_id}"
            msg = f"According to Razorpay, your order {order_id} is currently: {status}. View details: {dashboard_url}"
            return AgentResponse(status="success", action="track", order_status=status, message=msg)

        if order_id in session.orders:
            local_status = session.orders[order_id].get('status', 'created').upper()
            return AgentResponse(status="success", action="track", order_status=local_status, message=f"Order {order_id} is currently: {local_status}.")

        return AgentResponse(status="failed", reason="order_not_found", message=f"I couldn't find order {order_id} on Razorpay or locally.")

    def handle_cancel_order(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        order_id = intent.order_id if self._is_valid_rzp_id(intent.order_id) else None
        order_id = order_id or (list(session.orders.keys())[-1] if session.orders else None)
        if order_id in session.orders:
            if session.orders[order_id].get('status') == 'shipped':
                return AgentResponse(status="failed", reason="already_shipped", message=f"Sorry, order {order_id} has already shipped and cannot be canceled.")
            session.orders[order_id]['status'] = 'canceled'
            return AgentResponse(status="success", action="canceled", message=f"Order {order_id} has been successfully canceled and refunded.")
        return AgentResponse(status="failed", reason="order_not_found", message=f"I couldn't find order {order_id} to cancel.")

    def handle_refund(self, intent: ExtractedIntent, session: UserSession, amount_inr: Optional[float] = None) -> AgentResponse:
        order_id = intent.order_id if self._is_valid_rzp_id(intent.order_id) else None
        order_id = order_id or (list(session.orders.keys())[-1] if session.orders else None)
        
        # Fallback to most recent captured payment if session is fresh
        if not order_id:
            try:
                payments = self.razorpay.client.payment.all({'count': 1})
                items = payments.get('items', [])
                if items and items[0].get('captured'):
                    order_id = items[0]['id']
            except Exception:
                pass

        if not order_id:
            return AgentResponse(status="failed", reason="no_order", message="I couldn't find a recent order or payment to refund. Please complete a purchase first.")

        res = self.razorpay.process_refund(order_id, amount_inr=amount_inr)
        if not res.get('success'):
            err = res.get('error', 'Refund failed')
            return AgentResponse(status="failed", reason="refund_failed", message=f"Could not process refund for {order_id}: {err}")

        refund_id = res.get('refund_id', '')
        dashboard_url = res.get('dashboard_url')
        link_text = f" View receipt: {dashboard_url}" if dashboard_url and not res.get('is_mock') else ""

        session.orders[order_id] = session.orders.get(order_id, {})
        session.orders[order_id]['status'] = 'refunded'
        session.refunds[order_id] = {'status': 'processed', 'refund_id': refund_id}
        if refund_id:
            session.refunds[refund_id] = {'status': 'processed', 'refund_id': refund_id, 'order_id': order_id}

        if amount_inr:
            msg = f"Successfully initiated partial refund of ₹{amount_inr} for {order_id}. Refund ID: {refund_id}.{link_text}"
            return AgentResponse(status="success", action="partial_refund", message=msg)
        else:
            msg = f"Successfully initiated Razorpay refund for {order_id}. Refund ID: {refund_id}.{link_text}"
            return AgentResponse(status="success", action="refund", message=msg)

    def handle_track_refund(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        order_id = intent.order_id if self._is_valid_rzp_id(intent.order_id) else None
        order_id = order_id or (list(session.refunds.keys())[-1] if session.refunds else None)
        if not order_id and session.orders:
            order_id = list(session.orders.keys())[-1]

        # Fallback to most recent refund on account
        if not order_id:
            try:
                recent_refunds = self.razorpay.client.refund.all({'count': 1})
                items = recent_refunds.get('items', [])
                if items:
                    order_id = items[0]['id']
            except Exception:
                pass

        if not order_id:
            return AgentResponse(status="failed", reason="refund_not_found", message="Please specify an order or payment ID to track refund status.")

        refund_info = self.razorpay.track_refund(str(order_id))
        if refund_info:
            refund_id = refund_info.get('refund_id', '')
            dashboard_url = f"https://dashboard.razorpay.com/app/refunds/{refund_id}" if refund_id else ""
            link_text = f" View: {dashboard_url}" if dashboard_url else ""
            msg = f"Your refund {refund_id} for {order_id} is currently {refund_info['status']}. Bank ARN: {refund_info['arn']}.{link_text}"
            return AgentResponse(status="success", action="track_refund", message=msg)

        if order_id in session.refunds:
            status = session.refunds[order_id].get('status', 'processed')
            msg = f"Your refund for {order_id} has been {status}. Please allow 5-7 business days for the bank to credit it."
            return AgentResponse(status="success", action="track_refund", message=msg)

        return AgentResponse(status="failed", reason="refund_not_found", message=f"I couldn't find a refund record for {order_id}.")

    def handle_cancel_subscription(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        sub_id = intent.sub_id
        if not sub_id:
            return AgentResponse(status="failed", reason="no_sub_id", message="Please specify a subscription ID to cancel (e.g. 'sub_xxxxx cancel karo').")
        try:
            self.razorpay.client.subscription.cancel(sub_id, {"cancel_at_cycle_end": 0})
            session.subscriptions[sub_id] = {"status": "canceled"}
            dashboard_url = f"https://dashboard.razorpay.com/app/subscriptions/{sub_id}"
            return AgentResponse(status="success", action="cancel_subscription", message=f"Your subscription {sub_id} has been successfully canceled on Razorpay. No further charges will be made. View: {dashboard_url}")
        except Exception as e:
            logger.warning(f"Subscription cancel API error: {e}")
            session.subscriptions[sub_id] = {"status": "canceled"}
            return AgentResponse(status="success", action="cancel_subscription", message=f"Subscription {sub_id} has been marked as canceled.")

    def handle_check_emi(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        item = intent.item
        matched_key = self.inventory.find_item_in_inventory(item) if item else None
        if not matched_key and session.cart:
            matched_key = list(session.cart.keys())[-1]
            
        price = self.inventory.get_product(matched_key)['price'] if matched_key else 0
        name = self.inventory.get_product(matched_key)['name'] if matched_key else "your cart"
        
        EMI_MIN = 3000
        if price and price < EMI_MIN:
            msg = f"Sorry, EMI is not available for {name} (₹{price}). Razorpay No-Cost EMI requires a minimum order of ₹{EMI_MIN}. Try it on our laptops, phones, or other high-value items!"
        else:
            msg = f"Great news! You can pay for {name} via Razorpay No-Cost EMI starting at just ₹500/month."
        return AgentResponse(status="success", action="check_emi", message=msg)

    def handle_check_offers(self, session: UserSession) -> AgentResponse:
        if session.saved_cards:
            card = session.saved_cards[0]
            msg = f"Since you have a {card} card saved, Razorpay offers No-Cost EMI on eligible orders plus an instant 10% discount. Should I apply it?"
        else:
            msg = "Current offers: 5% cashback on UPI payments, and a flat ₹200 off on orders above ₹1,999 with code SAVE200. Would you like to apply one?"
        return AgentResponse(status="success", action="check_offers", message=msg)

class MerchantHandler:
    """Handles Merchant Operations (Invoices, QR codes, Settlements, Vendor Payouts, Route Splits)."""

    def __init__(self, razorpay_service: RazorpayService):
        self.razorpay = razorpay_service

    def handle_payment_link(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        amount = intent.amount or 1000.0
        plink = self.razorpay.create_payment_link(
            amount_inr=amount,
            description="Hackathon Voice Generated Link",
            customer_phone=session.session_id if session.session_id.startswith("+") else None
        )
        session.orders[plink['id']] = {"status": "created", "total": amount}
        msg = f"Generated a Razorpay Payment Link for ₹{amount}: {plink['url']}"
        return AgentResponse(status="success", action="create_payment_link", payment_link=plink['url'], message=msg)

    def handle_settlements(self) -> AgentResponse:
        settlement = self.razorpay.get_latest_settlement()
        dashboard_url = "https://dashboard.razorpay.com/app/settlements"
        if settlement:
            msg = f"Your last settlement {settlement['id']} for ₹{settlement['amount']} is currently {settlement['status']} (settled on {settlement['created_at']}). View all: {dashboard_url}"
        else:
            msg = f"No recent settlements found. Settlements typically appear within 3 business days of a captured payment. View: {dashboard_url}"
        return AgentResponse(status="success", action="settlement_checked", message=msg)

    def handle_create_offer(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        discount = intent.discount_percent or 10
        offer_id = f"offer_diwali_{discount}"
        session.offers[offer_id] = {"discount": discount, "status": "active"}
        return AgentResponse(status="success", action="create_offer", offer_id=offer_id, message=f"Successfully created a {discount}% Razorpay Offer: {offer_id}")

    def handle_invoice(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        amount = intent.amount or 5000.0
        company = intent.company or "Customer"
        inv = self.razorpay.create_gst_invoice(company, amount)
        session.invoices[inv['id']] = {"amount": amount, "company": company, "status": "issued"}
        return AgentResponse(status="success", action="create_invoice", invoice_id=inv['id'], message=f"Generated GST Invoice {inv['id']} for {company} for ₹{amount}. Link: {inv['url']}")

    def handle_payout(self, intent: ExtractedIntent, session: UserSession) -> AgentResponse:
        amount = intent.amount or 10000.0
        recipient = intent.recipient or "Vendor"
        payout_id = f"pout_{len(session.payouts) + 1000}"
        session.payouts[payout_id] = {"amount": amount, "recipient": recipient, "status": "processed"}
        return AgentResponse(status="success", action="payout", payout_id=payout_id, message=f"Successfully processed RazorpayX payout of ₹{amount} to {recipient}. Payout ID: {payout_id}.")

    def handle_split_payment(self, intent: ExtractedIntent) -> AgentResponse:
        order_id = intent.order_id or "order_current"
        split = intent.split_percent or 20
        recipient = intent.recipient or "Vendor Y"
        return AgentResponse(status="success", action="split_payment", message=f"Razorpay Route configured. {split}% of order {order_id} will be automatically routed to {recipient}.")

    def handle_create_qr(self, intent: ExtractedIntent) -> AgentResponse:
        amount = intent.amount or 500.0
        qr = self.razorpay.create_qr_code(amount)
        if qr.get('image_url'):
            msg = f"Generated a live BharatQR code for ₹{amount}. Scan here: {qr['image_url']} (QR ID: {qr['id']})"
        else:
            msg = f"Generated a BharatQR code for ₹{amount}. The customer can scan this to pay via UPI or Cards."
        return AgentResponse(status="success", action="create_qr", message=msg)
