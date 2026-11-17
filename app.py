import hashlib
import re

import streamlit as st
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.retrievers.bm25 import BM25Retriever

import readers
from local_embedding import LocalEmbedding

st.set_page_config(page_title="Chat with your data", page_icon="📄", layout="centered")

BOT, USER = "📄", "🙂"
MODELS = ["gemini-3.8-flash", "gemini-3.7-flash"]
PASSAGES = 8  # passages per question: enough to combine a document, a sheet and a slide
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


def friendly(error):
    text = str(error)
    if "API key not valid" in text or "API_KEY_INVALID" in text:
        return "That Gemini API key didn't work. Check it in the sidebar."
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "Gemini's free limit was reached. Wait a minute and try again."
    if "503" in text or "UNAVAILABLE" in text:
        return "Gemini is busy right now. Try again in a minute, or choose the other model."
    return f"Gemini returned an error: {text[:300]}"


SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def sentences(text):
    """Split text after . ! ? or … followed by a space. LlamaIndex's default splitter needs NLTK
    language data downloaded at run time, which fails on Streamlit Cloud."""
    parts, start = [], 0
    for match in SENTENCE_END.finditer(text):
        parts.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return parts


@st.cache_resource(show_spinner=False, max_entries=5)
def build_index(files):
    """Index the uploaded files. `files` is a tuple of (name, bytes), cached by content.
    Returns the search and messages about files that couldn't be read."""
    docs, problems = [], []
    for name, data in files:
        try:
            for part in readers.read(name, data):
                docs.append(Document(text=part["text"], metadata={"file_name": part["file"], "where": part["where"]}))
        except readers.Unreadable as error:
            problems.append(str(error))
    if not docs:
        return None, problems
    # passages of about 512 tokens: the embedding model reads 512, and a fact isn't drowned in a long passage
    index = VectorStoreIndex.from_documents(
        docs, embed_model=embedder(), transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64, chunking_tokenizer_fn=sentences)])
    return search(index), problems


def search(index):
    """Search by meaning (embeddings) and by words (BM25) together, merged by rank. Words catch exact
    names, codes and numbers that meaning search misses; meaning catches questions phrased differently."""
    nodes = list(index.docstore.docs.values())
    top = min(PASSAGES, len(nodes))
    return QueryFusionRetriever(
        [index.as_retriever(similarity_top_k=top), BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top)],
        similarity_top_k=top, num_queries=1, mode="reciprocal_rerank", use_async=False)


def source_list(nodes):
    seen, lines = set(), []
    for node in nodes:
        key = (node.metadata.get("file_name", "Document"), node.metadata.get("where", ""))
        if key not in seen:
            seen.add(key)
            lines.append(f"{key[0]}, {key[1]}" if key[1] else key[0])
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
    uploads = st.file_uploader("Your files", type=[ext.lstrip(".") for ext in readers.SUPPORTED],
                               accept_multiple_files=True,
                               help="Documents, spreadsheets, slides, PDFs, emails, ebooks, web pages and zip files.")
    model = st.selectbox("Model", MODELS)
    st.button("New chat", on_click=reset_chat, use_container_width=True)

st.title("Chat with your data")
st.markdown(
    "<div class='subtitle'>Upload documents and ask questions about them. "
    "Every answer shows where it came from.</div>",
    unsafe_allow_html=True,
)

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to start.")
    st.stop()
if not uploads:
    st.info("Upload one or more files in the sidebar: Word, Excel, PowerPoint, PDF, emails and more.")
    st.stop()

files = tuple(sorted((f.name, f.getvalue()) for f in uploads))
fingerprint = hashlib.sha256(b"".join(n.encode() + d for n, d in files) + model.encode()).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:  # new files or model: start over
    st.session_state.fingerprint = fingerprint
    reset_chat()

with st.spinner("Reading the documents..."):
    retriever, problems = build_index(files)
for problem in problems:
    st.warning(problem)
if retriever is None:
    st.stop()

if "chat_engine" not in st.session_state:
    try:
        llm = GoogleGenAI(model=model, api_key=api_key, temperature=0.3)
    except Exception as error:
        st.error(friendly(error))
        st.stop()
    st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever, llm=llm, system_prompt=SYSTEM_PROMPT)

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
