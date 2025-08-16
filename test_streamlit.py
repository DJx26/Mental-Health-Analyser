import streamlit as st
from chatbot import get_bot_response

st.title("🧠 Test Chatbot")

# Add a test section
st.header("Test Different Messages")

test_messages = [
    "I feel very sad and hopeless today",
    "I am so anxious about my presentation tomorrow",
    "I am feeling great and happy today", 
    "Work is stressing me out so much"
]

for i, message in enumerate(test_messages):
    if st.button(f"Test {i+1}: {message[:30]}..."):
        response = get_bot_response(message)
        st.write(f"**Input:** {message}")
        st.write(f"**Response:** {response}")
        st.divider()

# Add manual input
st.header("Manual Input")
user_input = st.text_input("Type your message:")
if st.button("Send") and user_input:
    response = get_bot_response(user_input)
    st.write(f"**Your message:** {user_input}")
    st.write(f"**Bot response:** {response}")

