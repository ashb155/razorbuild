import sys
import os

# Ensure the backend dir is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_pipeline import intent_pipeline

def test():
    print("\nTesting Intent Extraction...")
    queries = [
        "I want to buy a black jacket",
        "create a payment link for 2000",
        "refund my order 98765"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("Result:", intent_pipeline.extract_intent(q))

if __name__ == "__main__":
    test()
