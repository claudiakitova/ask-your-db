# Chat with your data

Upload one or more PDFs and ask questions about them. Every answer lists the pages it came from.

## How it works

1. You paste your own free Gemini API key in the sidebar. It stays in your browser session and isn't stored.
2. The app splits your PDFs into passages and indexes them with a small free model that runs inside the app, so searching costs nothing.
3. For each question, it finds the most relevant passages, and Gemini writes the answer from them, remembering the conversation.

Answers come in the language of the question, and the app says so when the documents don't cover something.

## Run it

1. Get a free Gemini API key at https://aistudio.google.com/apikey.
2. Install the requirements (Python 3.12):
   ```
   pip install -r requirements.txt
   ```
3. Start the app and paste your key in the sidebar:
   ```
   streamlit run app.py
   ```

## Tools

Python, Streamlit, LlamaIndex, Google Gemini, FastEmbed
