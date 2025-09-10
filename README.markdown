# EmpathyBot - Emotion-Aware Chatbot

## Overview
EmpathyBot is an AI-powered chatbot that detects emotions in user input and generates empathetic responses using a Retrieval-Augmented Generation (RAG) approach. Built with **Streamlit**, **DistilBERT** for emotion classification, and **FAISS** for efficient retrieval, this project combines natural language processing (NLP) and vector search to provide context-aware, emotionally intelligent responses.

## Features
- **Emotion Detection**: Uses `bhadresh-savani/distilbert-base-uncased-emotion` to classify emotions (joy, sadness, anger, fear, surprise, neutral) in user text.
- **Empathetic Responses**: Retrieves relevant response templates from a predefined corpus using `all-MiniLM-L6-v2` embeddings and FAISS for similarity search.
- **Interactive UI**: Streamlit-based interface with chat history, animated sidebar, and example prompts.
- **Disclaimer**: Includes a note encouraging users to seek professional help for serious issues, as the bot is not a therapist.

## Requirements
- Python 3.8+
- Libraries:
  ```bash
  pip install streamlit pandas numpy transformers sentence-transformers faiss-cpu
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/empathybot.git
   cd empathybot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Launch the app using the command above.
2. Enter a message in the text input field (e.g., "I just got a promotion and I’m so happy! 🎉").
3. Click **Submit** to receive an empathetic response based on the detected emotion.
4. View the chat history and retrieved response templates in the UI.

## Example
**Input**: "I’m feeling really down today because of work 😞"  
**Output**:  
- Emotion: sadness (score: 0.95)  
- Response: "That sounds really tough — I'm here to listen if you want to share more.  
  (Disclaimer: I'm not a therapist; please seek professional help for serious issues.)"

## Project Structure
- `app.py`: Main Streamlit application script.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file.

## How It Works
1. **Emotion Classification**:
   - Uses DistilBERT to predict the dominant emotion in the user's input.
   - Returns the emotion label and confidence score.
2. **Response Retrieval**:
   - Encodes user input and corpus texts using `all-MiniLM-L6-v2`.
   - Performs similarity search with FAISS to retrieve top-k relevant response templates.
   - Prioritizes templates matching the detected emotion.
3. **UI Rendering**:
   - Displays chat history with styled user and bot messages.
   - Shows retrieved templates in an expandable section.
   - Includes an animated sidebar with random greetings and example prompts.

## Limitations
- The bot is not a substitute for professional mental health support.
- Emotion detection accuracy depends on the DistilBERT model and input clarity.
- The corpus is predefined and may not cover all scenarios.
- FAISS index is built in-memory and not persisted.

## Future Improvements
- Expand the response corpus for more diverse replies.
- Persist the FAISS index for faster startup.
- Add support for multilingual emotion detection.
- Integrate real-time feedback to refine responses.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/), [Hugging Face Transformers](https://huggingface.co/), [Sentence Transformers](https://sbert.net/), and [FAISS](https://github.com/facebookresearch/faiss).
- Emotion model: `bhadresh-savani/distilbert-base-uncased-emotion`.
- Embedding model: `all-MiniLM-L6-v2`.