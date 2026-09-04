import time
from collections import defaultdict
from typing import Dict, Any, Optional, List

class UserSession:
    """Encapsulates all session state for a single user/device."""
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.cart: Dict[str, int] = {}
        self.orders: Dict[str, Any] = {}
        self.refunds: Dict[str, Any] = {}
        self.subscriptions: Dict[str, Any] = {}
        self.invoices: Dict[str, Any] = {}
        self.payouts: Dict[str, Any] = {}
        self.saved_cards: List[str] = []
        self.offers: Dict[str, Any] = {}
        
        # Dialog state for conversational follow-ups
        self.pending_search: Optional[str] = None
        self.pending_cross_sell: Optional[str] = None
        self.pending_alternative: Optional[str] = None
        self.pending_confirmation: Optional[Dict[str, Any]] = None
        
        # Fraud rate limiting timestamps
        self.action_timestamps: List[float] = []

    def is_rate_limited(self, max_requests: int = 15, window_seconds: int = 60) -> bool:
        """Checks if session exceeded financial request rate limit (Thirdwatch simulation)."""
        now = time.time()
        self.action_timestamps = [t for t in self.action_timestamps if now - t < window_seconds]
        if len(self.action_timestamps) >= max_requests:
            return True
        self.action_timestamps.append(now)
        return False

    def clear_pending_follow_ups(self):
        """Clears stale follow-up states when a new distinct action occurs."""
        self.pending_search = None
        self.pending_cross_sell = None
        self.pending_alternative = None

class SessionManager:
    """Manages multi-tenant in-memory sessions (can be backed by Redis in production)."""
    def __init__(self):
        self._sessions: Dict[str, UserSession] = {}

    def get_session(self, session_id: Optional[str]) -> UserSession:
        sid = session_id if session_id else "guest_default_session"
        if sid not in self._sessions:
            self._sessions[sid] = UserSession(sid)
        return self._sessions[sid]

# Singleton instance
session_manager = SessionManager()
