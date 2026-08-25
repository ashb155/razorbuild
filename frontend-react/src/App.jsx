import React, { useEffect, useState } from 'react';
import './App.css';
import RazorpayAgent from './components/RazorpayAgent';
import { ShoppingCart, Plus, X } from 'lucide-react';
import axios from 'axios';

function App() {
  const [inventory, setInventory] = useState({});
  const [cart, setCart] = useState({});
  const [loading, setLoading] = useState(true);

  const fetchCart = () => {
    axios.get('http://127.0.0.1:8000/api/cart')
      .then(res => {
        if (res.data) {
          setCart(res.data);
        }
      })
      .catch(err => {
        console.error("Network error loading cart", err);
      });
  };

  useEffect(() => {
    // Fetch inventory
    axios.get('http://127.0.0.1:8000/api/inventory')
      .then(res => {
        if (res.data && !res.data.error) {
          setInventory(res.data);
        } else {
          console.error("Failed to load inventory", res.data.error);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Network error loading inventory", err);
        setLoading(false);
      });

    // Fetch initial cart
    fetchCart();
  }, []);

  const totalCartItems = Object.values(cart).reduce((sum, qty) => sum + qty, 0);

  return (
    <div>
      <header className="header">
        <div className="logo">Nexus Store <span className="premium-tag">PREMIUM</span></div>
        <div className="nav-links">
          <span>Electronics</span>
          <span>Footwear</span>
          <span>Accessories</span>
          <span>Support</span>
          <div className="cart-icon-container">
            <ShoppingCart size={24} color="#2d2a26" />
            {totalCartItems > 0 && (
              <span className="cart-badge">{totalCartItems}</span>
            )}
          </div>
        </div>
      </header>

      <section className="hero">
        <h1>Discover Premium Tech & Lifestyle</h1>
        <p>Experience seamless, conversational shopping powered by Voice AI. Tap the microphone to talk to our agent!</p>
      </section>

      <main className="inventory-grid">
        {loading ? (
          <h2>Loading Premium Catalog...</h2>
        ) : (
          Object.values(inventory).map((item) => (
            <div key={item.id} className={`product-card ${item.stock === 0 ? 'out-of-stock' : ''}`}>
              <div className="image-container">
                <img 
                  src={item.image || "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&q=80&w=300"} 
                  alt={item.name} 
                  className="product-image" 
                />
                {item.stock === 0 && <div className="sold-out-overlay">SOLD OUT</div>}
              </div>
              <div className="product-info">
                <h3 className="product-name">{item.name}</h3>
                <div className="price-row">
                  <div className="product-price">₹{item.price.toLocaleString()}</div>
                  <button 
                    className="add-icon-btn" 
                    disabled={item.stock === 0}
                    title={item.stock > 0 ? "Add to Cart" : "Out of Stock"}
                  >
                    {item.stock > 0 ? <Plus size={20} /> : <X size={20} />}
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </main>

      <RazorpayAgent onCartUpdate={fetchCart} />
    </div>
  );
}

export default App;
