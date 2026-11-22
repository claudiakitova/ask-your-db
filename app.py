import hashlib

import streamlit as st

import ai
import readers
import search

st.set_page_config(page_title="Ask your data", page_icon="icon.png", layout="centered")

BOT, USER = ":material/plagiarism:", ":material/person:"
PASSAGES = 8  # passages per question: enough to combine a document, a sheet and a slide

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
    return search.embedder()


@st.cache_resource(show_spinner=False, max_entries=5)
def build_index(files):
    """Read and index the uploaded files. `files` is a tuple of (name, bytes), cached by content.
    Returns the index and messages about files that couldn't be read."""
    parts, problems = [], []
    for name, data in files:
        try:
            parts += readers.read(name, data)
        except readers.Unreadable as error:
            problems.append(str(error))
    return (search.Index(parts, embedder()) if parts else None), problems


def source_list(passages):
    lines = []
    for p in passages:
        line = f"{p['file']}, {p['where']}" if p["where"] else p["file"]
        if line not in lines:
            lines.append(line)
    return lines


def show_message(message):
    with st.chat_message(message["role"], avatar=BOT if message["role"] == "assistant" else USER):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                st.markdown("<div class='source'>" + "<br>".join(message["sources"]) + "</div>",
                            unsafe_allow_html=True)


def history():
    """The conversation so far, without the greeting and error messages."""
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            if m.get("chat", True)]


def reset_chat():
    st.session_state.messages = []


with st.sidebar:
    st.markdown("### :material/plagiarism: Ask your data")
    service = st.selectbox("AI service", list(ai.SERVICES))
    key_url = ai.SERVICES[service]["key_url"]
    api_key = st.text_input(
        f"{service} API key", type="password", key=f"key_{service}",
        help=f"Free at {key_url.split('//')[1]}. It stays in your browser session and isn't stored.",
    )
    st.caption(f"[Get a free key]({key_url})")
    uploads = st.file_uploader("Your files",  # no type list: readers.read explains any file it can't read
                               accept_multiple_files=True,
                               help="Documents, spreadsheets, slides, PDFs, emails, ebooks, web pages and zip files.")
    model = st.selectbox("Model", ai.SERVICES[service]["models"], key=f"model_{service}")
    st.button("New chat", on_click=reset_chat, use_container_width=True)

st.title("Ask your data")
st.markdown(
    "<div class='subtitle'>Upload documents and ask questions about them. "
    "Every answer shows where it came from.</div>",
    unsafe_allow_html=True,
)

if not api_key:
    st.info(f"Enter your {service} API key in the sidebar to start. Gemini, Mistral and Groq all give free keys.")
    st.stop()
if not uploads:
    st.info("Upload one or more files in the sidebar: Word, Excel, PowerPoint, PDF, emails and more.")
    st.stop()

files = tuple(sorted((f.name, f.getvalue()) for f in uploads))
fingerprint = hashlib.sha256(b"".join(n.encode() + d for n, d in files)).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:  # new files: start over
    st.session_state.fingerprint = fingerprint
    reset_chat()

with st.spinner("Reading the documents..."):
    index, problems = build_index(files)
for problem in problems:
    st.warning(problem)
if index is None:
    st.stop()

if not st.session_state.messages:
    names = ", ".join(n for n, _ in files)
    st.session_state.messages = [{"role": "assistant", "content": f"Ready. Ask me anything about {names}.",
                                  "chat": False}]

for message in st.session_state.messages:
    show_message(message)

if prompt := st.chat_input("Ask about your documents"):
    earlier = history()
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    show_message(user_message)
    with st.chat_message("assistant", avatar=BOT):
        with st.spinner("Searching the documents..."):
            try:
                question = ai.standalone(service, model, api_key, earlier, prompt)
                passages = index.search(question, ai.SERVICES[service].get("passages", PASSAGES))
                reply = {"role": "assistant", "content": ai.answer(service, model, api_key, earlier, prompt, passages),
                         "sources": source_list(passages)}
            except ai.AIError as error:
                reply = {"role": "assistant", "content": str(error), "chat": False}
                user_message["chat"] = False  # an unanswered question isn't sent along later
    st.session_state.messages.append(reply)
    st.rerun()
