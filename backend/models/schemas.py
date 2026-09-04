from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class IntentAction(str, Enum):
    # Search & Catalog
    SEARCH = "search"
    PRICE_CHECK = "price_check"
    FIND = "find"
    QUERY = "query"
    
    # Cart & Purchase
    ADD = "add"
    ADD_TO_CART = "add_to_cart"
    BUY = "buy"
    PURCHASE = "purchase"
    CHECKOUT = "checkout"
    CONFIRM_CART = "confirm_cart"
    REMOVE = "remove"
    DELETE = "delete"
    REMOVE_FROM_CART = "remove_from_cart"
    REMOVE_ITEM = "remove_item"
    
    # Customer Order & Account
    TRACK = "track"
    CHECK_STATUS = "check_status"
    ORDER_STATUS = "order_status"
    CANCEL = "cancel"
    CANCEL_ORDER = "cancel_order"
    PAY = "pay"
    MAKE_PAYMENT = "make_payment"
    REFUND = "refund"
    GET_REFUND = "get_refund"
    PARTIAL_REFUND = "partial_refund"
    TRACK_REFUND = "track_refund"
    CANCEL_SUBSCRIPTION = "cancel_subscription"
    STOP_SUBSCRIPTION = "stop_subscription"
    HANDLE_FAILED_PAYMENT = "handle_failed_payment"
    MAGIC_CHECKOUT_ADDRESS = "magic_checkout_address"
    CHECK_OFFERS = "check_offers"
    CHECK_EMI = "check_emi"
    SAVE_CARD = "save_card"
    
    # Merchant & Operations
    CREATE_PAYMENT_LINK = "create_payment_link"
    PAYMENT_LINK = "payment_link"
    GENERATE_PAYMENT_LINK = "generate_payment_link"
    CHECK_SETTLEMENT = "check_settlement"
    SETTLEMENT_STATUS = "settlement_status"
    CHECK_SETTLEMENTS = "check_settlements"
    CREATE_OFFER = "create_offer"
    CREATE_DISCOUNT = "create_discount"
    CREATE_INVOICE = "create_invoice"
    INVOICE = "invoice"
    GENERATE_INVOICE = "generate_invoice"
    PAYOUT = "payout"
    SEND_PAYOUT = "send_payout"
    TRANSFER = "transfer"
    SPLIT_PAYMENT = "split_payment"
    SPLIT = "split"
    ROUTE = "route"
    CREATE_QR = "create_qr"
    
    # FAQ / Support
    FAQ = "faq"
    QUESTION = "question"
    HELP = "help"
    SUPPORT = "support"
    
    UNKNOWN = "unknown"

class ExtractedIntent(BaseModel):
    action: str = Field(..., description="Action name extracted from user utterance")
    item: Optional[str] = Field(None, description="Product or entity name")
    quantity: int = Field(1, description="Quantity requested")
    size: Optional[str] = Field(None, description="Product size if specified")
    order_id: Optional[str] = Field(None, description="Order or payment identifier")
    amount: Optional[float] = Field(None, description="Monetary amount in INR")
    sub_id: Optional[str] = Field(None, description="Subscription identifier")
    topic: Optional[str] = Field(None, description="Support or FAQ topic")
    discount_percent: Optional[int] = Field(None, description="Discount percentage for offers")
    company: Optional[str] = Field(None, description="Customer or company name for invoices")
    recipient: Optional[str] = Field(None, description="Recipient name/account for payouts")
    split_percent: Optional[int] = Field(None, description="Split percentage for route transfers")
    card_network: Optional[str] = Field(None, description="Card network name for tokenization")
    requires_confirmation: bool = Field(False, description="Whether this action requires OTP/money movement gate")

class Product(BaseModel):
    id: str
    name: str
    price: float
    stock: int
    category: str
    image: Optional[str] = None

class AgentResponse(BaseModel):
    status: str = Field("success", description="Status code: success, failed, pending_confirmation, blocked, etc.")
    message: str = Field(..., description="Conversational text response to be displayed & spoken")
    action: Optional[str] = Field(None, description="Action executed")
    payment_link: Optional[str] = Field(None, description="Short URL for Razorpay payment link")
    order_status: Optional[str] = Field(None, description="Status of tracked order")
    invoice_id: Optional[str] = Field(None, description="Generated invoice identifier")
    payout_id: Optional[str] = Field(None, description="Generated payout identifier")
    offer_id: Optional[str] = Field(None, description="Generated offer identifier")
    alternative: Optional[str] = Field(None, description="Suggested alternative product key")
    available: Optional[int] = Field(None, description="Available stock quantity")
    reason: Optional[str] = Field(None, description="Failure reason code")
    raw_intent: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, omitting None fields for clean API JSON responses."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
