from prediction import predict_condition, friendly_message_for

# Test different types of messages
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

print("Testing current model predictions:\n")
for i, message in enumerate(test_messages, 1):
    condition, confidence = predict_condition(message)
    response = friendly_message_for(condition)
    print(f"Test {i}: '{message}'")
    print(f"  → Condition: {condition} (confidence: {confidence:.3f})")
    print(f"  → Response: {response}")
    print()

