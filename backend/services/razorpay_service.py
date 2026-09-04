import os
import time
import logging
from typing import Dict, Any, Optional
import razorpay

logger = logging.getLogger(__name__)

class RazorpayService:
    """Service wrapping Razorpay SDK APIs with graceful fallback modes for live demos."""

    def __init__(self):
        key_id = os.environ.get('RAZORPAY_KEY_ID', '')
        key_secret = os.environ.get('RAZORPAY_KEY_SECRET', '')
        self.client = razorpay.Client(auth=(key_id, key_secret))
        self.is_configured = bool(key_id and key_secret)

    def create_payment_link(self, amount_inr: float, description: str, customer_phone: Optional[str] = None) -> Dict[str, Any]:
        """Creates a Razorpay standard payment link."""
        amount_paise = int(amount_inr * 100)
        phone = customer_phone or "+919999999999"
        
        try:
            link_data = {
                "amount": amount_paise,
                "currency": "INR",
                "description": description,
                "customer": {
                    "name": "Voice Assistant Customer",
                    "contact": phone,
                    "email": "customer@example.com"
                },
                "notify": {"sms": False, "email": False}
            }
            plink = self.client.payment_link.create(link_data)
            return {
                "id": plink.get('id'),
                "url": plink.get('short_url'),
                "is_mock": False
            }
        except Exception as e:
            logger.warning(f"Razorpay Payment Link API error, using mock fallback: {e}")
            mock_id = f"plink_mock_{int(time.time())}"
            mock_url = f"https://rzp.io/i/{mock_id}"
            return {
                "id": mock_id,
                "url": mock_url,
                "is_mock": True
            }

    def fetch_order_status(self, order_id: str) -> Optional[str]:
        """Fetches status of a live Razorpay order, payment, or payment link."""
        try:
            if str(order_id).startswith('pay_'):
                payment = self.client.payment.fetch(str(order_id))
                raw = payment.get('status', 'created')
                status_map = {
                    'captured': 'PAID',
                    'refunded': 'REFUNDED',
                    'failed': 'FAILED',
                    'created': 'PENDING',
                    'authorized': 'AUTHORIZED',
                }
                return status_map.get(raw, raw.upper())
            elif str(order_id).startswith('plink_'):
                plink = self.client.payment_link.fetch(str(order_id))
                raw = plink.get('status', 'created')
                status_map = {
                    'paid': 'PAID',
                    'partially_paid': 'PARTIALLY PAID',
                    'created': 'PENDING (Payment Link not yet paid)',
                    'cancelled': 'CANCELLED',
                    'expired': 'EXPIRED',
                }
                return status_map.get(raw, raw.upper())
            elif str(order_id).startswith('order_'):
                real_order = self.client.order.fetch(str(order_id))
                return real_order.get('status', 'created').upper()
            else:
                rzp_orders = self.client.order.fetch_all({"receipt": str(order_id)})
                if rzp_orders.get('items'):
                    return rzp_orders['items'][0].get('status', 'created').upper()
        except Exception as e:
            logger.warning(f"Failed to fetch order status from Razorpay: {e}")
        return None

    def _resolve_payment_id(self, target_id: str) -> Optional[str]:
        """Resolves a real payment_id (pay_...) from a payment, payment link (plink_), order (order_), or receipt."""
        if not target_id:
            return None
        target = str(target_id).strip()
        if target.startswith('pay_'):
            return target

        if target.startswith('plink_'):
            try:
                plink = self.client.payment_link.fetch(target)
                payments = plink.get('payments') or []
                if payments:
                    return payments[-1].get('payment_id') or payments[-1].get('id')
                order_id = plink.get('order_id')
                if order_id:
                    order_payments = self.client.order.payments(order_id)
                    items = order_payments.get('items', [])
                    if items:
                        return items[0].get('id')
            except Exception as e:
                logger.warning(f"Failed resolving payment from payment link {target}: {e}")

        if target.startswith('order_'):
            try:
                payments = self.client.order.payments(target)
                items = payments.get('items', [])
                if items:
                    return items[0].get('id')
            except Exception as e:
                logger.warning(f"Failed resolving payment from order {target}: {e}")

        try:
            rzp_orders = self.client.order.fetch_all({"receipt": target})
            items = rzp_orders.get('items', [])
            if items:
                order_id = items[0].get('id')
                payments = self.client.order.payments(order_id)
                p_items = payments.get('items', [])
                if p_items:
                    return p_items[0].get('id')
        except Exception as e:
            logger.warning(f"Failed resolving payment from receipt {target}: {e}")

        try:
            recent_payments = self.client.payment.all({"count": 1})
            items = recent_payments.get('items', [])
            if items and items[0].get('status') == 'captured':
                return items[0].get('id')
        except Exception as e:
            logger.warning(f"Failed resolving recent captured payment: {e}")

        return None

    def process_refund(self, order_or_payment_id: str, amount_inr: Optional[float] = None) -> Dict[str, Any]:
        """Initiates full or partial live refund on Razorpay."""
        try:
            payment_id = self._resolve_payment_id(order_or_payment_id)
            if not payment_id:
                return {"success": False, "error": f"No captured payment found for {order_or_payment_id}"}

            refund_data = {"speed": "optimum"}
            if amount_inr is not None:
                refund_data["amount"] = int(amount_inr * 100)

            res = self.client.payment.refund(payment_id, refund_data)
            refund_id = res.get('id', 'rfnd_live')
            return {
                "success": True,
                "refund_id": refund_id,
                "payment_id": payment_id,
                "dashboard_url": f"https://dashboard.razorpay.com/app/refunds/{refund_id}",
                "is_mock": False
            }
        except Exception as e:
            logger.warning(f"Razorpay Refund API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "is_mock": False
            }

    def track_refund(self, order_or_payment_id: str) -> Optional[Dict[str, Any]]:
        """Tracks the refund status and ARN for a given payment ID, refund ID, or order/link on Razorpay."""
        try:
            target = str(order_or_payment_id).strip()
            if target.startswith('rfnd_'):
                ref = self.client.refund.fetch(target)
                return {
                    "refund_id": ref.get('id'),
                    "status": ref.get('status', 'processed'),
                    "arn": ref.get('acquirer_data', {}).get('arn') or 'Pending'
                }

            payment_id = self._resolve_payment_id(target)
            if not payment_id:
                return None

            rzp_refunds = self.client.refund.all({"payment_id": payment_id})
            items = rzp_refunds.get('items', [])
            if items:
                latest = items[0]
                return {
                    "refund_id": latest.get('id'),
                    "status": latest.get('status', 'processed'),
                    "arn": latest.get('acquirer_data', {}).get('arn') or 'Pending'
                }
        except Exception as e:
            logger.warning(f"Failed to track refund on Razorpay: {e}")
        return None

    def get_latest_settlement(self) -> Optional[Dict[str, Any]]:
        """Queries the merchant's latest settlement batch."""
        try:
            settlements = self.client.settlement.all({"count": 1})
            items = settlements.get('items', [])
            if items:
                latest = items[0]
                return {
                    "id": latest.get('id'),
                    "amount": latest.get('amount', 0) / 100.0,
                    "status": latest.get('status'),
                    "created_at": time.strftime('%Y-%m-%d', time.localtime(latest.get('created_at', time.time())))
                }
        except Exception as e:
            logger.warning(f"Failed to fetch settlements: {e}")
        return None

    def create_gst_invoice(self, company: str, amount_inr: float) -> Dict[str, Any]:
        """Generates a B2B GST-compliant Invoice."""
        try:
            invoice_data = {
                "type": "invoice",
                "description": f"Invoice for {company}",
                "customer": {"name": company},
                "line_items": [{
                    "name": "Services",
                    "amount": int(amount_inr * 100),
                    "currency": "INR",
                    "quantity": 1
                }]
            }
            invoice = self.client.invoice.create(invoice_data)
            return {
                "id": invoice.get('id'),
                "url": invoice.get('short_url'),
                "is_mock": False
            }
        except Exception as e:
            logger.warning(f"Invoice API error, using mock fallback: {e}")
            inv_id = f"inv_mock_{int(time.time())}"
            return {
                "id": inv_id,
                "url": f"https://rzp.io/i/{inv_id}",
                "is_mock": True
            }

    def create_qr_code(self, amount_inr: float) -> Dict[str, Any]:
        """Creates a BharatQR code for the given amount using the Razorpay API."""
        import time as _time
        close_by = int(_time.time()) + 3600  # expires in 1 hour
        try:
            qr = self.client.qr_code.create({
                "type": "bharat_qr",
                "name": "Voice Commerce Payment",
                "usage": "single_use",
                "fixed_amount": True,
                "payment_amount": int(amount_inr * 100),
                "description": f"Payment of \u20b9{amount_inr} via Voice Assistant",
                "close_by": close_by,
            })
            return {
                "id": qr.get('id'),
                "image_url": qr.get('image_url'),
                "is_mock": False
            }
        except Exception as e:
            logger.warning(f"QR Code API error, using mock fallback: {e}")
            return {"id": None, "image_url": None, "is_mock": True}

# Singleton instance
razorpay_service = RazorpayService()
