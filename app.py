import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI

from local_embedding import LocalEmbedding

st.set_page_config(page_title="Chat with your data", page_icon="📄", layout="centered")

BOT, USER = "📄", "🙂"
MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
SYSTEM_PROMPT = (
    "You answer questions about the documents the user uploaded. Use only what the "
    "documents say. Get straight to the answer, without greetings. If the documents "
    "don't cover something, say so instead of guessing. Answer in the language of the question."
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.5rem; max-width: 760px;}
    .subtitle {color: #9AA4B8; font-size: 1.05rem; margin-top: -0.8rem; margin-bottom: 1.5rem;}
    .source {color: #9AA4B8; font-size: 0.9rem; line-height: 1.6;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def embedder():
    return LocalEmbedding()


Settings.embed_model = embedder()  # never fall back to OpenAI defaults
Settings.llm = None


@st.cache_resource(show_spinner=False, max_entries=5)
def build_index(files):
    """Index the uploaded PDFs. `files` is a tuple of (name, bytes), cached by content."""
    with tempfile.TemporaryDirectory() as folder:
        for name, data in files:
            (Path(folder) / name).write_bytes(data)
        docs = SimpleDirectoryReader(folder, required_exts=[".pdf"]).load_data()
    return VectorStoreIndex.from_documents(docs, embed_model=embedder())


def source_list(nodes):
    seen, lines = set(), []
    for node in nodes:
        key = (node.metadata.get("file_name", "Document"), node.metadata.get("page_label"))
        if key not in seen:
            seen.add(key)
            lines.append(f"{key[0]}, p. {key[1]}" if key[1] else key[0])
    return lines


def show_message(message):
    with st.chat_message(message["role"], avatar=BOT if message["role"] == "assistant" else USER):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                st.markdown("<div class='source'>" + "<br>".join(message["sources"]) + "</div>",
                            unsafe_allow_html=True)


def reset_chat():
    st.session_state.messages = []
    st.session_state.pop("chat_engine", None)


with st.sidebar:
    st.markdown("### 📄 Chat with your data")
    api_key = st.text_input(
        "Gemini API key", type="password",
        help="Free at aistudio.google.com/apikey. It stays in your browser session and isn't stored.",
    )
    st.caption("[Get a free key](https://aistudio.google.com/apikey)")
    uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    model = st.selectbox("Model", MODELS)
    st.button("New chat", on_click=reset_chat, use_container_width=True)

st.title("Chat with your data")
st.markdown(
    "<div class='subtitle'>Upload documents and ask questions about them. "
    "Every answer shows the pages it came from.</div>",
    unsafe_allow_html=True,
)

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to start.")
    st.stop()
if not uploads:
    st.info("Upload one or more PDF files in the sidebar.")
    st.stop()

files = tuple(sorted((f.name, f.getvalue()) for f in uploads))
fingerprint = hashlib.sha256(b"".join(n.encode() + d for n, d in files) + model.encode()).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:  # new files or model: start over
    st.session_state.fingerprint = fingerprint
    reset_chat()

with st.spinner("Reading the documents..."):
    index = build_index(files)

def friendly(error):
    text = str(error)
    if "API key not valid" in text or "API_KEY_INVALID" in text:
        return "That Gemini API key didn't work. Check it in the sidebar."
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "Gemini's free limit was reached. Wait a minute and try again."
    return f"Gemini returned an error: {text[:300]}"


if "chat_engine" not in st.session_state:
    try:
        llm = GoogleGenAI(model=model, api_key=api_key, temperature=0.3)
    except Exception as error:
        st.error(friendly(error))
        st.stop()
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", system_prompt=SYSTEM_PROMPT, llm=llm
    )

if not st.session_state.messages:
    names = ", ".join(n for n, _ in files)
    st.session_state.messages = [{"role": "assistant", "content": f"Ready. Ask me anything about {names}."}]

for message in st.session_state.messages:
    show_message(message)

if prompt := st.chat_input("Ask about your documents"):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    show_message(user_message)
    with st.chat_message("assistant", avatar=BOT):
        with st.spinner("Searching the documents..."):
            try:
                response = st.session_state.chat_engine.chat(prompt)
                answer = {"role": "assistant", "content": response.response,
                          "sources": source_list(response.source_nodes)}
            except Exception as error:  # bad key, quota, network
                answer = {"role": "assistant", "content": friendly(error)}
    st.session_state.messages.append(answer)
    st.rerun()
