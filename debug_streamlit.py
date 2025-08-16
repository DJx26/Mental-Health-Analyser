from chatbot import get_bot_response

# Test the exact function that Streamlit uses
test_messages = [
    "I feel very sad and hopeless today",
    "I am so anxious about my presentation tomorrow", 
    "I am feeling great and happy today",
    "Work is stressing me out so much",
    "I feel lonely and empty inside",
    "I'm worried about everything",
    "Today was amazing and I feel wonderful",
    "I'm so frustrated with my job"
]

print("Testing get_bot_response function (what Streamlit uses):\n")
for i, message in enumerate(test_messages, 1):
    response = get_bot_response(message)
    print(f"Test {i}: '{message}'")
    print(f"  → Response: {response}")
    print("-" * 80)
    print()

