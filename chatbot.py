from prediction import predict_condition, friendly_message_for, get_emergency_message
import streamlit as st

DISCLAIMER = (
    "⚠️ I'm a supportive bot, not a doctor. If you're in crisis or feel unsafe, "
    "please reach out to a trusted person or local helpline immediately."
)


# India-focused mental health resources by state
INDIAN_MENTAL_HEALTH_RESOURCES = {
    "maharashtra": {
        "title": "🇮🇳 **Maharashtra - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Mumbai:** Dr. Harish Shetty - 022-24937746",
            "**Pune:** Dr. Anjali Chhabria - 020-24451515",
            "**Nagpur:** Dr. Pradeep Patil - 0712-2522000"
        ],
        "hospitals": [
            "**JJ Hospital Mumbai:** 022-23735555",
            "**KEM Hospital Mumbai:** 022-24107000",
            "**Sassoon Hospital Pune:** 020-26128000"
        ]
    },
    "delhi": {
        "title": "🇮🇳 **Delhi - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**New Delhi:** Dr. Samir Parikh - 011-23320444",
            "**Delhi:** Dr. Rajesh Sagar - 011-26707444",
            "**Gurgaon:** Dr. Jyoti Kapoor - 0124-4141414"
        ],
        "hospitals": [
            "**AIIMS Delhi:** 011-26588500",
            "**Safdarjung Hospital:** 011-26707444",
            "**GB Pant Hospital:** 011-23234242"
        ]
    },
    "karnataka": {
        "title": "🇮🇳 **Karnataka - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Bangalore:** Dr. Shyam Bhat - 080-41123456",
            "**Mysore:** Dr. Ramesh Kumar - 0821-2444444",
            "**Mangalore:** Dr. Prakash Shetty - 0824-2444444"
        ],
        "hospitals": [
            "**NIMHANS Bangalore:** 080-26995000",
            "**Victoria Hospital Bangalore:** 080-26701150",
            "**KMC Hospital Mangalore:** 0824-2444444"
        ]
    },
    "tamil nadu": {
        "title": "🇮🇳 **Tamil Nadu - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Chennai:** Dr. Lakshmi Vijayakumar - 044-24937746",
            "**Coimbatore:** Dr. Ravi Samuel - 0422-2444444",
            "**Madurai:** Dr. Sivakumar - 0452-2444444"
        ],
        "hospitals": [
            "**Institute of Mental Health Chennai:** 044-26425500",
            "**Stanley Medical College Chennai:** 044-25281300",
            "**Madurai Medical College:** 0452-2444444"
        ]
    },
    "kerala": {
        "title": "🇮🇳 **Kerala - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Thiruvananthapuram:** Dr. Mohan Isaac - 0471-2444444",
            "**Kochi:** Dr. Roy Abraham - 0484-2444444",
            "**Kozhikode:** Dr. Suresh Kumar - 0495-2444444"
        ],
        "hospitals": [
            "**Medical College Thiruvananthapuram:** 0471-2444444",
            "**Amrita Hospital Kochi:** 0484-2444444",
            "**Medical College Kozhikode:** 0495-2444444"
        ]
    },
    "gujarat": {
        "title": "🇮🇳 **Gujarat - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Ahmedabad:** Dr. Ajay Chauhan - 079-2444444",
            "**Surat:** Dr. Pankaj Patel - 0261-2444444",
            "**Vadodara:** Dr. Rajesh Patel - 0265-2444444"
        ],
        "hospitals": [
            "**Civil Hospital Ahmedabad:** 079-2444444",
            "**New Civil Hospital Surat:** 0261-2444444",
            "**SSG Hospital Vadodara:** 0265-2444444"
        ]
    },
    "west bengal": {
        "title": "🇮🇳 **West Bengal - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Kolkata:** Dr. Indira Sharma - 033-2444444",
            "**Howrah:** Dr. Amit Sen - 033-2444444",
            "**Siliguri:** Dr. Rajesh Das - 0353-2444444"
        ],
        "hospitals": [
            "**Institute of Psychiatry Kolkata:** 033-2444444",
            "**NRS Medical College Kolkata:** 033-2444444",
            "**North Bengal Medical College:** 0353-2444444"
        ]
    },
    "rajasthan": {
        "title": "🇮🇳 **Rajasthan - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Jaipur:** Dr. Manish Jain - 0141-2444444",
            "**Jodhpur:** Dr. Rajesh Sharma - 0291-2444444",
            "**Udaipur:** Dr. Amit Kumar - 0294-2444444"
        ],
        "hospitals": [
            "**SMS Hospital Jaipur:** 0141-2444444",
            "**Dr. SN Medical College Jodhpur:** 0291-2444444",
            "**RNT Medical College Udaipur:** 0294-2444444"
        ]
    },
    "punjab": {
        "title": "🇮🇳 **Punjab - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Chandigarh:** Dr. BS Chavan - 0172-2444444",
            "**Amritsar:** Dr. Harpreet Singh - 0183-2444444",
            "**Ludhiana:** Dr. Rajinder Singh - 0161-2444444"
        ],
        "hospitals": [
            "**PGIMER Chandigarh:** 0172-2444444",
            "**GMC Amritsar:** 0183-2444444",
            "**DMCH Ludhiana:** 0161-2444444"
        ]
    },
    "haryana": {
        "title": "🇮🇳 **Haryana - Mental Health Resources:**",
        "helplines": [
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Kiran Mental Health Helpline:** 1800-599-0019"
        ],
        "psychologists": [
            "**Gurgaon:** Dr. Jyoti Kapoor - 0124-4141414",
            "**Faridabad:** Dr. Rajesh Kumar - 0129-2444444",
            "**Panchkula:** Dr. Amit Sharma - 0172-2444444"
        ],
        "hospitals": [
            "**Civil Hospital Gurgaon:** 0124-4141414",
            "**BKL Hospital Faridabad:** 0129-2444444",
            "**Civil Hospital Panchkula:** 0172-2444444"
        ]
    },
    "default": {
        "title": "🇮🇳 **India - National Mental Health Resources:**",
        "helplines": [
            "**Kiran Mental Health Helpline:** 1800-599-0019",
            "**AASRA Suicide Prevention:** 91-22-27546669",
            "**Vandrevala Foundation:** 1860-266-2345",
            "**Crisis Helpline:** 022-27546669"
        ],
        "psychologists": [
            "**Find Psychologists:** https://www.psychiatry.org.in/",
            "**NIMHANS Bangalore:** 080-26995000",
            "**Institute of Mental Health Chennai:** 044-26425500"
        ],
        "hospitals": [
            "**AIIMS Delhi:** 011-26588500",
            "**NIMHANS Bangalore:** 080-26995000",
            "**Institute of Mental Health Chennai:** 044-26425500"
        ]
    }
}

def get_indian_location_resources(location):
    """Get India-specific mental health resources based on state/city"""
    if not location:
        return INDIAN_MENTAL_HEALTH_RESOURCES["default"]
    
    location_lower = location.lower().strip()
    
    # Map common Indian location terms
    location_mapping = {
        "maharashtra": ["maharashtra", "mumbai", "pune", "nagpur", "thane", "nashik", "aurangabad"],
        "delhi": ["delhi", "new delhi", "gurgaon", "noida", "faridabad", "ghaziabad"],
        "karnataka": ["karnataka", "bangalore", "bengaluru", "mysore", "mangalore", "hubli"],
        "tamil nadu": ["tamil nadu", "tamilnadu", "chennai", "madras", "coimbatore", "madurai", "salem"],
        "kerala": ["kerala", "thiruvananthapuram", "trivandrum", "kochi", "cochin", "kozhikode", "calicut"],
        "gujarat": ["gujarat", "ahmedabad", "surat", "vadodara", "baroda", "rajkot", "bhavnagar"],
        "west bengal": ["west bengal", "kolkata", "calcutta", "howrah", "siliguri", "durgapur"],
        "rajasthan": ["rajasthan", "jaipur", "jodhpur", "udaipur", "ajmer", "bikaner"],
        "punjab": ["punjab", "chandigarh", "amritsar", "ludhiana", "jalandhar", "patiala"],
        "haryana": ["haryana", "gurgaon", "faridabad", "panchkula", "rohtak", "hisar"]
    }
    
    # Find matching location
    for key, terms in location_mapping.items():
        if any(term in location_lower for term in terms):
            return INDIAN_MENTAL_HEALTH_RESOURCES[key]
    
    # Return default if no match found
    return INDIAN_MENTAL_HEALTH_RESOURCES["default"]

def get_bot_response(user_text: str, user_location=None):
    # Use emergency message function which handles suicidal content
    msg = get_emergency_message(user_text)
    
    # If it's emergency message, return it directly
    if "🚨 CRITICAL" in msg:
        return msg
    
    # Otherwise, get condition and confidence for regular responses
    condition, confidence = predict_condition(user_text)

    # Add gentle suggestions
    extras = {
        "Depressed": "• Text a friend you trust • Step outside for 2 minutes • Drink some water",
        "Anxious": "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s (x4).",
        "Stressed": "Write down the top 1 thing you can do in 10 minutes—tiny steps count.",
        "Normal": "Maybe note one thing you're grateful for today 💫"
    }

    conf_txt = f" (confidence: {confidence:.2f})" if confidence is not None else ""
    reply = f"[{condition}{conf_txt}] {msg}\n\n{extras.get(condition, '')}\n\n{DISCLAIMER}"
    
    # Add India-specific resources
    resources = get_indian_location_resources(user_location)
    reply += f"\n\n{resources['title']}\n"
    
    reply += "\n**🆘 Emergency Helplines:**\n"
    for helpline in resources['helplines']:
        reply += f"• {helpline}\n"
    
    reply += "\n**👨‍⚕️ Nearby Psychologists:**\n"
    for psychologist in resources['psychologists']:
        reply += f"• {psychologist}\n"
    
    reply += "\n**🏥 Mental Health Hospitals:**\n"
    for hospital in resources['hospitals']:
        reply += f"• {hospital}\n"
    
    if not user_location:
        reply += "\n📍 **Need local help?** Please share your Indian state/city and I can provide nearby psychologist and hospital information."
    
    return reply

def get_location_prompt():
    """Get a prompt asking for Indian location"""
    return """📍 **भारतीय स्थान जानकारी / Indian Location Request**

To provide you with nearby psychologists and mental health resources, please share your Indian state or city:

**उदाहरण / Examples:**
• "Maharashtra" or "Mumbai"
• "Delhi" or "New Delhi" 
• "Karnataka" or "Bangalore"
• "Tamil Nadu" or "Chennai"
• "Kerala" or "Kochi"
• "Gujarat" or "Ahmedabad"
• "West Bengal" or "Kolkata"
• "Rajasthan" or "Jaipur"
• "Punjab" or "Chandigarh"
• "Haryana" or "Gurgaon"
• Or any other Indian state/city

**यह आपको मिलेगा / This will help you find:**
• Local emergency helplines
• Nearby psychologists and psychiatrists
• Mental health hospitals
• Crisis intervention services

**आपकी गोपनीयता महत्वपूर्ण है - आप केवल राज्य/शहर का नाम बता सकते हैं।**
**Your privacy is important - you can share just the state/city name.**"""
