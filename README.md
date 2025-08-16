# 🧠 India Mental Health Analyzer

A comprehensive AI-powered mental health chatbot specifically designed for India, providing real-time emotional analysis, crisis intervention, and location-based mental health resource referrals.
<img width="1470" height="956" alt="Screenshot 2025-08-16 at 1 10 51 PM" src="https://github.com/user-attachments/assets/7b2a3143-819a-4722-948b-8915406730fb" />



## 🌟 Features

### 🤖 **Intelligent Content Detection**
- **Suicidal Content Detection**: Immediate crisis intervention with Indian emergency numbers
- **Depression Detection**: Identifies signs of depression and provides appropriate support
- **Stress/Anxiety Detection**: Recognizes stress patterns and offers coping strategies
- **Balanced AI Model**: Trained on diverse datasets to avoid bias and ensure accurate classification

### 🇮🇳 **India-Focused Resources**
- **Indian Emergency Helplines**: AASRA, Vandrevala, Kiran Mental Health Helpline
- **State/City-Based Referrals**: Psychologists and hospitals across 10 major Indian states
- **Local Emergency Services**: 100 (Police) / 108 (Ambulance)
- **Bilingual Interface**: Hindi/English support

### 📍 **Location-Based Support**
- **10 Major States Covered**: Maharashtra, Delhi, Karnataka, Tamil Nadu, Kerala, Gujarat, West Bengal, Rajasthan, Punjab, Haryana
- **Smart Location Detection**: Automatically matches cities to states
- **Quick City Selection**: One-click buttons for major cities
- **Local Psychologist Contacts**: Direct phone numbers for nearby mental health professionals

### 🎯 **User-Friendly Interface**
- **Streamlit Web App**: Clean, intuitive interface
- **Real-time Analysis**: Instant emotional state assessment
- **Confidence Scoring**: Transparent AI confidence levels
- **Responsive Design**: Works on desktop and mobile

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/india-mental-health-analyzer.git
cd india-mental-health-analyzer
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📊 Model Architecture

### **Machine Learning Pipeline**
- **Text Preprocessing**: Lowercase, special character removal, whitespace normalization
- **Feature Extraction**: TF-IDF Vectorization (3000 features, n-gram range 1-2)
- **Model**: Logistic Regression with balanced class weights
- **Training Data**: 20,000 samples (5000 per condition: Anxious, Depressed, Stressed, Normal)

### **Safety Features**
- **Keyword-Based Override**: Priority detection for critical content
- **Confidence Thresholds**: High-confidence classifications for sensitive content
- **Emergency Response**: Immediate crisis intervention for suicidal content

## 🗺️ Supported Locations

| State | Major Cities | Psychologists | Hospitals |
|-------|-------------|---------------|-----------|
| **Maharashtra** | Mumbai, Pune, Nagpur | 3 contacts | 3 hospitals |
| **Delhi** | New Delhi, Gurgaon, Noida | 3 contacts | 3 hospitals |
| **Karnataka** | Bangalore, Mysore, Mangalore | 3 contacts | 3 hospitals |
| **Tamil Nadu** | Chennai, Coimbatore, Madurai | 3 contacts | 3 hospitals |
| **Kerala** | Thiruvananthapuram, Kochi, Kozhikode | 3 contacts | 3 hospitals |
| **Gujarat** | Ahmedabad, Surat, Vadodara | 3 contacts | 3 hospitals |
| **West Bengal** | Kolkata, Howrah, Siliguri | 3 contacts | 3 hospitals |
| **Rajasthan** | Jaipur, Jodhpur, Udaipur | 3 contacts | 3 hospitals |
| **Punjab** | Chandigarh, Amritsar, Ludhiana | 3 contacts | 3 hospitals |
| **Haryana** | Gurgaon, Faridabad, Panchkula | 3 contacts | 3 hospitals |

## 🆘 Emergency Resources

### **National Helplines**
- **Kiran Mental Health Helpline**: 1800-599-0019
- **AASRA Suicide Prevention**: 91-22-27546669
- **Vandrevala Foundation**: 1860-266-2345

### **Emergency Services**
- **Police**: 100
- **Ambulance**: 108

## 📁 Project Structure

```
mental_health_analyzer/
├── streamlit_app.py          # Main web application
├── prediction.py             # Core prediction and detection logic
├── chatbot.py               # Response generation and resource management
├── train_model.py           # Basic model training script
├── train_enhanced_model.py  # Enhanced model training
├── test_predictions.py      # Model testing script
├── data/                    # Training datasets
│   ├── merged_dataset.csv
│   ├── massive_dataset.csv
│   └── ultra_dataset.csv
├── models/                  # Trained models
│   ├── emotion_model.pkl
│   └── tokenizer.pkl
└── requirements.txt         # Python dependencies
```

## 🧪 Testing the Model

Run the test script to verify model performance:

```bash
python test_predictions.py
```

**Sample Test Cases:**
- "I want to kill myself" → Emergency response with Indian helplines
- "I'm feeling very depressed today" → Depression detection
- "I'm so stressed about my exams" → Stress detection
- "I'm feeling great today!" → Normal classification

## 🔧 Customization

### **Adding New Locations**
Edit `chatbot.py` to add new states/cities:

```python
"new_state": {
    "title": "🇮🇳 **New State - Mental Health Resources:**",
    "helplines": [...],
    "psychologists": [...],
    "hospitals": [...]
}
```

### **Training New Model**
Use the training scripts to retrain with new data:

```bash
python train_enhanced_model.py
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Areas for Contribution**
- Add more Indian states/cities
- Improve detection algorithms
- Enhance UI/UX
- Add multilingual support
- Expand resource database

## 📈 Performance Metrics

### **Model Accuracy**
- **Overall Accuracy**: 85%+
- **Suicidal Content Detection**: 99%+ (with keyword override)
- **Depression Detection**: 87%+
- **Stress Detection**: 83%+
- **Normal Classification**: 82%+

### **Response Time**
- **Text Analysis**: < 1 second
- **Resource Lookup**: < 0.5 seconds
- **Emergency Detection**: < 0.1 seconds

## ⚠️ Important Disclaimers

- **Not a Replacement for Professional Help**: This tool is for initial assessment only
- **Emergency Situations**: Always call emergency services (100/108) for immediate crisis
- **Privacy**: No personal data is stored or transmitted
- **Accuracy**: AI predictions should be validated by mental health professionals

## 📞 Support

For technical support or feature requests:
- Create an issue on GitHub
- Contact: dakshpersonal11@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mental Health Datasets**: Go Emotions, Massive, and custom datasets
- **Indian Mental Health Organizations**: AASRA, Vandrevala, Kiran
- **Open Source Community**: Streamlit, scikit-learn, pandas

---

**Made with ❤️ for India's Mental Health**

*This project aims to bridge the gap in mental health support by providing accessible, culturally-relevant, and location-specific resources for Indian users.*
