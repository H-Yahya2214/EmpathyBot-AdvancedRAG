# 🤖 Empathetic AI Chatbot

An advanced conversational AI system that combines emotion detection, empathetic response generation, and speech capabilities to create meaningful, supportive interactions.

## 🌟 Features

- **🎭 Advanced Emotion Detection**: Uses DistilBERT to identify 6+ emotions with confidence scores
- **💬 Empathetic Response Generation**: RAG-powered system with 30+ contextual response templates
- **⚡ Semantic Search**: FAISS vector search with sentence transformers for relevant response retrieval
- **🎤 Full Speech Support**: Speech-to-text input and text-to-speech output
- **📊 Conversation Analytics**: Real-time emotion tracking and conversation statistics
- **🎨 Modern UI**: Beautiful Streamlit interface with custom CSS and responsive design

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or create the project:**
```bash
mkdir empathetic-chatbot
cd empathetic-chatbot
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open your browser** to `http://localhost:8501`

## 📂 Project Structure

```
empathetic-chatbot/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
├── .streamlit_cache/          # Cached models and data
│   ├── faiss_index.index     # FAISS vector index
│   └── faiss_index_embeddings.npy  # Precomputed embeddings
└── empathetic_corpus.json     # Response templates (auto-generated)
```

## 🛠️ Technical Architecture

### Core Components

1. **Emotion Detection Pipeline**
   - Model: `bhadresh-savani/distilbert-base-uncased-emotion`
   - Detects: joy, sadness, anger, fear, surprise, love, neutral
   - Real-time classification with confidence scores

2. **RAG System (Retrieval-Augmented Generation)**
   - Embeddings: `all-MiniLM-L6-v2` sentence transformer
   - Vector Store: FAISS with cosine similarity
   - Response Templates: 30+ empathetic responses per emotion

3. **Speech Processing**
   - STT: Google Speech Recognition API
   - TTS: Google Text-to-Speech (gTTS)
   - Audio Format: WAV input, MP3 output

4. **Web Interface**
   - Framework: Streamlit with custom CSS
   - Features: Real-time chat, voice recording, emotion visualization
   - Responsive design with sidebar controls

### Data Flow

```
User Input → Emotion Detection → RAG Retrieval → Response Generation → TTS Output
     ↓              ↓                 ↓               ↓              ↓
  Text/Voice → DistilBERT → FAISS Search → Template → Audio
```

## 🎯 Usage Guide

### Text Chat
1. Type your message in the text area
2. Click "Send Message" to get an empathetic response
3. View detected emotion and confidence score

### Voice Chat  
1. Click the microphone button to record
2. Speak your message clearly
3. Click "
