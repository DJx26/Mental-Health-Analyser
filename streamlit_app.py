import streamlit as st
from chatbot import get_bot_response, get_location_prompt
import os

st.set_page_config(page_title="Friendly Mental Health Chatbot", page_icon="🧠", layout="centered")
st.title("🧠 Friendly Mental Health Chatbot")
st.caption("Share how you're feeling. I'll try to understand and respond supportively.\n\n"
           "⚠️ Not a medical device; for emergencies contact local services.")

# Add a refresh button to clear cache and reload model
if st.button("🔄 Refresh Model"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Model refreshed! Try sending a new message.")
    st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_location" not in st.session_state:
    st.session_state.user_location = None

def add_message(sender, text):
    st.session_state.messages.append({"sender": sender, "text": text})

# Location input section
st.sidebar.header("📍 भारतीय स्थान / Indian Location")
st.sidebar.markdown("Share your Indian state/city to get nearby psychologists and mental health resources.")

location_input = st.sidebar.text_input(
    "Your Indian State/City:",
    value=st.session_state.user_location or "",
    placeholder="e.g., Maharashtra, Mumbai, Delhi, Bangalore..."
)

if st.sidebar.button("Set Location"):
    if location_input.strip():
        st.session_state.user_location = location_input.strip()
        st.sidebar.success(f"Location set to: {location_input.strip()}")
        st.rerun()
    else:
        st.sidebar.error("Please enter an Indian state or city.")

if st.session_state.user_location:
    st.sidebar.markdown(f"**Current Location:** {st.session_state.user_location}")
    if st.sidebar.button("Clear Location"):
        st.session_state.user_location = None
        st.rerun()

# Quick location buttons for common Indian cities
st.sidebar.markdown("**Quick Select:**")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Mumbai"):
        st.session_state.user_location = "Mumbai"
        st.rerun()
    if st.button("Delhi"):
        st.session_state.user_location = "Delhi"
        st.rerun()
    if st.button("Bangalore"):
        st.session_state.user_location = "Bangalore"
        st.rerun()
    if st.button("Chennai"):
        st.session_state.user_location = "Chennai"
        st.rerun()
    if st.button("Kolkata"):
        st.session_state.user_location = "Kolkata"
        st.rerun()

with col2:
    if st.button("Pune"):
        st.session_state.user_location = "Pune"
        st.rerun()
    if st.button("Ahmedabad"):
        st.session_state.user_location = "Ahmedabad"
        st.rerun()
    if st.button("Jaipur"):
        st.session_state.user_location = "Jaipur"
        st.rerun()
    if st.button("Kochi"):
        st.session_state.user_location = "Kochi"
        st.rerun()
    if st.button("Chandigarh"):
        st.session_state.user_location = "Chandigarh"
        st.rerun()

# Main chat interface
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("Type your message", "")
    submitted = st.form_submit_button("Send")
    
    if submitted and user_text.strip():
        add_message("You", user_text.strip())
        
        # Get bot response with location
        bot_reply = get_bot_response(user_text.strip(), st.session_state.user_location)
        add_message("Bot", bot_reply)

# Display chat messages
for m in st.session_state.messages:
    if m["sender"] == "You":
        st.markdown(f"**You:** {m['text']}")
    else:
        st.markdown(f"**Bot:** {m['text']}")

# Add a clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Location prompt if no location is set
if not st.session_state.user_location:
    st.info(get_location_prompt())