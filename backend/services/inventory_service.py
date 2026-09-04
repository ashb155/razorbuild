import os
import json
import difflib
from typing import Dict, Optional, Tuple, List
from models.schemas import Product

class InventoryService:
    """Service handling product catalog lookup, fuzzy name resolution, and recommendations."""

    # Default complementary cross-sell map (category -> recommended complementary category)
    DEFAULT_CROSS_SELL_MATRIX = {
        'watch': 'headphones',
        'smartphone': 'headphones',
        'shoes': 'clothing',
        'clothing': 'shoes',
        'laptop': 'accessories',
        'accessories': 'laptop',
        'kitchen': 'food',
        'food': 'kitchen',
        'cosmetics': 'bag',
        'bag': 'glasses',
        'glasses': 'clothing'
    }

    def __init__(self, catalog_path: Optional[str] = None):
        if not catalog_path:
            catalog_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inventory.json")
        self.catalog_path = catalog_path
        self._inventory: Dict[str, Dict] = {}
        self.load_inventory()

    def load_inventory(self) -> Dict[str, Dict]:
        """Loads and caches the inventory catalog from disk."""
        if os.path.exists(self.catalog_path):
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                self._inventory = json.load(f)
        return self._inventory

    def get_raw_inventory(self) -> Dict[str, Dict]:
        return self._inventory

    def get_product(self, item_key: str) -> Optional[Dict]:
        return self._inventory.get(item_key)

    def find_item_in_inventory(self, query: str) -> Optional[str]:
        """
        Multi-tier fuzzy resolution against human-readable product names and keys.
        Includes a word-overlap guard to prevent prefix/brand collisions (e.g. 'boat airdopes' -> 'boat cable').
        """
        if not query:
            return None
            
        q_lower = query.strip().lower()

        # 1. Exact key match
        if q_lower in self._inventory:
            return q_lower

        name_to_key = {v['name'].lower(): k for k, v in self._inventory.items()}

        def has_word_overlap(text_a: str, text_b: str, min_len: int = 4) -> bool:
            words_a = {w for w in text_a.split() if len(w) >= min_len}
            words_b = {w for w in text_b.split() if len(w) >= min_len}
            return bool(words_a & words_b)

        # 2. Close match against official product names (boAt Airdopes 141, Noise ColorFit Pro 4, etc.)
        name_matches = difflib.get_close_matches(q_lower, name_to_key.keys(), n=3, cutoff=0.45)
        for m in name_matches:
            if has_word_overlap(q_lower, m):
                return name_to_key[m]

        # 3. Close match against underscore keys (noise_smartwatch, etc.)
        key_matches = difflib.get_close_matches(q_lower, self._inventory.keys(), n=3, cutoff=0.55)
        for k in key_matches:
            if has_word_overlap(q_lower, k.replace('_', ' ')):
                return k

        # 4. Token overlap fallback (handles multi-word English/Transliterated product names)
        tokens = [t for t in q_lower.split() if len(t) >= 4]
        for token in tokens:
            for k, v in self._inventory.items():
                if token in v['name'].lower() or token in k:
                    return k

        return None

    def suggest_alternative(self, out_of_stock_key: str) -> Optional[str]:
        """Finds an in-stock alternative in the exact same product category."""
        item_data = self.get_product(out_of_stock_key)
        if not item_data:
            return None
            
        category = item_data.get('category')
        for k, data in self._inventory.items():
            if k != out_of_stock_key and data.get('category') == category and data.get('stock', 0) > 0:
                return k
        return None

    def suggest_any_alternative(self, search_term: str) -> Optional[str]:
        """Fallback recommendation when the requested term does not exist in catalog at all."""
        # Loose match
        loose = difflib.get_close_matches(search_term.lower(), self._inventory.keys(), n=1, cutoff=0.2)
        if loose and self._inventory[loose[0]].get('stock', 0) > 0:
            return loose[0]
            
        # Word match
        words = search_term.lower().split()
        for w in words:
            for k, data in self._inventory.items():
                if w in k and data.get('stock', 0) > 0:
                    return k
                    
        # General top in-stock product
        for k, data in self._inventory.items():
            if data.get('stock', 0) > 0:
                return k
        return None

    def get_cross_sell(self, item_key: str, cart: Optional[Dict[str, int]] = None) -> Optional[str]:
        """Suggests a complementary product based on category graph, skipping items already in user's cart."""
        item_data = self.get_product(item_key)
        if not item_data:
            return None
            
        cart = cart or {}
        cat = item_data.get('category')
        target_cat = self.DEFAULT_CROSS_SELL_MATRIX.get(cat, cat)
        
        for k, v in self._inventory.items():
            if k != item_key and k not in cart and v.get('category') == target_cat and v.get('stock', 0) > 0:
                return v['name']
        return None
