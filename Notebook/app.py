import streamlit as st
import random
import librosa
import soundfile as sf
from gtts import gTTS
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import whisper
from st_audiorec import st_audiorec
import numpy as np

# -----------------------
# 0) Helper: Audio Preprocessing
# -----------------------
def preprocess_audio(in_path, out_path="proc.wav", sr=16000, top_db=20):
    """
    Load audio, convert to mono, trim silence, normalize, and save as 16-bit WAV.
    """
    y, _ = librosa.load(in_path, sr=sr, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    y_norm = y_trim / np.max(np.abs(y_trim)) if np.max(np.abs(y_trim)) > 0 else y_trim
    sf.write(out_path, y_norm, sr, subtype='PCM_16')
    return out_path

# -----------------------
# 1) Load Emotion Classifier (DistilBERT)
# -----------------------
EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    return tokenizer, model

tokenizer_em, model_em = load_model()
em_pipeline = pipeline("text-classification", model=model_em, tokenizer=tokenizer_em, return_all_scores=True)

def predict_emotion(text):
    """
    Predict emotion label and confidence score from text.
    """
    out = em_pipeline(text)[0]
    best = max(out, key=lambda x: x["score"])
    return best["label"].lower(), best["score"]

# -----------------------
# 2) RAG Corpus + FAISS for Empathetic Responses
# -----------------------
corpus = [
    {"emotion": "joy", "text": "That's wonderful to hear — what made you feel this way?"},
    {"emotion": "joy", "text": "Amazing! I'd love to hear more about what's bringing you joy."},
    {"emotion": "joy", "text": "Sounds fantastic! What was the highlight of your day?"},
    {"emotion": "joy", "text": "Wow, that’s really exciting. Want to share more details?"},
    {"emotion": "sadness", "text": "I'm so sorry you're feeling down. Do you want to tell me what's going on?"},
    {"emotion": "sadness", "text": "That sounds really tough — I'm here to listen if you want to share more."},
    {"emotion": "sadness", "text": "I hear your sadness. Sometimes talking about it can make it lighter."},
    {"emotion": "sadness", "text": "I can imagine this is hard. Do you want some comfort or advice?"},
    {"emotion": "anger", "text": "I can hear how upset you are. Do you want to talk about what happened?"},
    {"emotion": "anger", "text": "It's understandable to feel angry about that — what's the worst part for you?"},
    {"emotion": "anger", "text": "That sounds really frustrating. How are you handling it right now?"},
    {"emotion": "anger", "text": "Your anger makes sense. Would you like to vent more about it?"},
    {"emotion": "fear", "text": "That sounds scary. Are you safe right now? Would you like to talk it through?"},
    {"emotion": "fear", "text": "I hear your fear. What's the part that worries you the most?"},
    {"emotion": "fear", "text": "It must feel overwhelming. Do you want to think through ways to feel safer?"},
    {"emotion": "fear", "text": "Thanks for sharing this. What could help reduce your fear right now?"},
    {"emotion": "surprise", "text": "Oh — that's surprising. How do you feel about that?"},
    {"emotion": "surprise", "text": "Wow, I didn’t expect that! What was your first reaction?"},
    {"emotion": "surprise", "text": "That must have caught you off guard. Do you see it as good or bad?"},
    {"emotion": "surprise", "text": "That's unexpected. Do you want to explore what this means for you?"},
    {"emotion": "neutral", "text": "Thanks for sharing that — want to tell me more?"},
    {"emotion": "neutral", "text": "I understand. Can you explain it in a bit more detail?"},
    {"emotion": "neutral", "text": "Got it. How does this situation affect you personally?"},
    {"emotion": "neutral", "text": "Okay, interesting. What do you think is the next step for you?"}
]

@st.cache_resource
def load_rag():
    """
    Load sentence embeddings and FAISS index for RAG retrieval.
    """
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in corpus]
    embs = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return embed_model, index

embed_model, index = load_rag()

def retrieve_templates(user_text, detected_emotion, top_k=3):
    """
    Retrieve top-k empathetic responses based on user text and emotion.
    """
    q_emb = embed_model.encode([user_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0:
            continue
        row = corpus[idx]
        same_em = (row["emotion"].lower() == detected_emotion.lower())
        results.append({"text": row["text"], "emotion": row["emotion"], "same_emotion": same_em})
    return sorted(results, key=lambda x: (not x["same_emotion"]))[:top_k]

def few_shot_reply(user_text, detected_emotion):
    """
    Generate reply based on RAG templates and add disclaimer.
    """
    templates = retrieve_templates(user_text, detected_emotion)
    base = templates[0]["text"] if templates else "Thanks for sharing. Do you want to tell me more?"
    reply = f"{base}\n\n(Disclaimer: I'm not a therapist; please seek professional help for serious issues.)"
    return reply, templates

# -----------------------
# 3) Speech-to-Text: Whisper
# -----------------------
@st.cache_resource
def load_whisper_model(size="small"):
    return whisper.load_model(size)

DEFAULT_WHISPER_SIZE = "small"

def transcribe_whisper(path, whisper_size=DEFAULT_WHISPER_SIZE, language=None):
    """
    Transcribe audio file to text using Whisper.
    """
    model = load_whisper_model(whisper_size)
    opts = {}
    if language:
        opts['language'] = language
        opts['task'] = 'transcribe'
    result = model.transcribe(path, **opts)
    return result.get('text', '').strip()

# -----------------------
# 4) Text-to-Speech (TTS)
# -----------------------
def text_to_speech(text, filename="tts_output.mp3"):
    """
    Convert text to speech and save as MP3.
    """
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# ===============================
# 5) Streamlit UI
# ===============================
st.set_page_config(page_title="EmpathyBot Chat", page_icon="🤖", layout="wide")
st.title("💬 EmpathyBot - Emotion-Aware Chat 🤖")
st.write("This demo uses DistilBERT for emotion detection, RAG for empathetic responses, and Whisper for speech-to-text.")

# -----------------------
# Sidebar: Greetings and Examples
# -----------------------
greetings = [
    "Hey there! 👋 How are you feeling today?",
    "Welcome! 😊 I'm here to listen.",
    "Hello! 🌟 Share your thoughts with me."
]
st.sidebar.markdown(f"<p style='color:#666;'>{random.choice(greetings)}</p>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <p style='text-align:center; font-size:30px;'>
        <span style='display:inline-block; animation:bounce 1s infinite;'>💬</span>
    </p>
    <style>
    @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color:#424242;'>Examples of What You Can Write or Say:</h4>", unsafe_allow_html=True)

examples = [
    "I just got a promotion and I’m so happy! 🎉",
    "I’m feeling really down today because of work 😞",
    "This traffic is making me so angry! 😡",
    "I’m scared about the upcoming exam 😨",
    "That news caught me by surprise 😲",
    "Just sharing my day—what do you think? 🤔"
]
st.sidebar.markdown(f"<ul style='color:#666; padding-left:20px;'><li>{random.choice(examples)}</li></ul>", unsafe_allow_html=True)
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align:center; color:#424242;'>🤖 About EmpathyBot</h3>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<p style='color:#666;'>EmpathyBot is an AI-powered chatbot designed to detect emotions in your text or speech and respond with empathetic, context-aware messages. It uses advanced natural language processing to identify feelings like joy, sadness, anger, fear, surprise, and neutral tones.</p>",
    unsafe_allow_html=True
)

# -----------------------
# Initialize chat history
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# User Input: Text
# -----------------------
user_input = st.text_input("✍️ Type your message:", placeholder="Share your thoughts...")

# -----------------------
# User Input: Record Audio
# -----------------------
st.markdown("#### 🎤  Record your voice (click record, then stop):")
wav_audio_data = st_audiorec()
if wav_audio_data:
    raw_path = "recorded_raw.wav"
    with open(raw_path, "wb") as f:
        f.write(wav_audio_data)
    st.audio(raw_path, format='audio/wav')

    y, sr = librosa.load(raw_path, sr=None, mono=True)
    dur, peak = len(y)/sr, float(np.max(np.abs(y)))
    st.write(f"Recording duration: {dur:.2f}s — peak amplitude: {peak:.3f}")

    proc_path = preprocess_audio(raw_path, out_path="recorded_proc.wav")
    st.write("Preprocessed audio saved.")

    user_input = transcribe_whisper(proc_path, whisper_size='small', language='english')
    st.success(f"Transcribed Speech: {user_input}")

# -----------------------
# Send Button (Custom Style)
# -----------------------
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #111;
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    border: 2px solid #333;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #333;
    transform: scale(1.05);
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

if st.button(" Submit"):
    if user_input.strip():
        pred_label, score = predict_emotion(user_input)
        reply, templates = few_shot_reply(user_input, pred_label)

        # Add chat messages to session state
        st.session_state.chat_history.append({"role": "user", "content": user_input, "time": datetime.now().strftime("%H:%M")})
        st.session_state.chat_history.append({
            "role": "bot",
            "content": f"[Emotion: {pred_label}, {score:.2f}] → {reply}",
            "templates": [t["text"] for t in templates],
            "time": datetime.now().strftime("%H:%M")
        })

        # Play TTS audio
        tts_file = text_to_speech(reply)
        st.audio(tts_file, format="audio/mp3", autoplay=True)
    else:
        st.warning("Please enter a message before submitting.")

# -----------------------
# Display Chat History
# -----------------------
st.markdown("### Conversation")
chat_container = st.container(border=True)
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='text-align:right; color:white; background:#111; padding:12px; "
                f"border-radius:10px; margin:8px; max-width:70%; float:right; clear:both;'>"
                f"<b>You:</b> {msg['content']}<br><small>{msg['time']}</small></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align:left; color:black; background:#eceff1; padding:12px; "
                f"border-radius:10px; margin:8px; max-width:70%; float:left; clear:both;'>"
                f"<b>Bot:</b> {msg['content']}<br><small>{msg['time']}</small></div>",
                unsafe_allow_html=True
            )
            with st.expander("Retrieved Responses"):
                for i, t in enumerate(msg.get("templates", []), 1):
                    st.write(f"Response {i}: {t}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("<center><sub>⚡ Powered by DistilBERT + RAG + Whisper | Built with Streamlit</sub></center>", unsafe_allow_html=True)
