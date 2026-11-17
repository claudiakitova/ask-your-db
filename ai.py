"""Step 3 of RAG (retrieval-augmented generation): an AI writes the answer from the passages.

This is the "augmented generation" part: the AI gets the question together with the passages
search.py found, and is told to answer from those passages only. That's what keeps it from
making things up, and why every answer can list its sources. The AI never sees the rest of
your files, only these few passages.

Gemini, Mistral and Groq all accept the same OpenAI-style request, so one function talks to
all three. This is the only step that sends anything over the internet."""

import requests

# The services and models offered in the sidebar. Each has a free key, and each takes requests
# at its own web address.
SERVICES = {
    "Gemini": {"models": ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"],
               "key_url": "https://aistudio.google.com/apikey",
               "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
    "Mistral": {"models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
                "key_url": "https://console.mistral.ai/api-keys",
                "url": "https://api.mistral.ai/v1/chat/completions"},
    # Groq's free plan allows 8,000 tokens a minute, so it gets fewer passages per question
    "Groq": {"models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"], "passages": 5,
             "key_url": "https://console.groq.com/keys",
             "url": "https://api.groq.com/openai/v1/chat/completions"},
}
# The instructions sent with every question. "Use only what the documents say" and "say so
# instead of guessing" are the core of RAG: the answer comes from your files, not from what the
# model learned elsewhere.
SYSTEM_PROMPT = (
    "You answer questions about the documents the user uploaded. Use only what the "
    "documents say. Get straight to the answer, without greetings. If the documents "
    "don't cover something, say so instead of guessing. Answer in the language of the question."
)
# Used before searching, to turn a follow-up into a question that stands on its own (see standalone).
REWRITE_PROMPT = (
    "Rewrite the last question so it can be understood without the conversation before it. "
    "Keep its language. Reply with the rewritten question only."
)
# How much of the conversation goes along with each request. More gives the AI context but
# costs tokens on every question.
HISTORY = 10  # earlier messages sent along, so follow-up questions make sense


# Any failure (bad key, limit reached, service busy) becomes a sentence the app can show as is.
class AIError(Exception):
    """The message is ready to show to the user."""


# The one call to the internet. It sends the messages to the chosen service and returns the text
# it writes back. Temperature 0.3 keeps answers close to the documents rather than inventive.
def ask(service, model, key, messages):
    try:
        response = requests.post(SERVICES[service]["url"], headers={"Authorization": f"Bearer {key}"},
                                 json={"model": model, "messages": messages, "temperature": 0.3}, timeout=120)
    except requests.RequestException:
        raise AIError(f"Couldn't reach {service}. Check the connection and try again.")
    if response.status_code != 200:
        raise AIError(explain(service, response.status_code, response.text))
    content = response.json()["choices"][0]["message"].get("content")
    if isinstance(content, list):  # some models send their reasoning too; keep only the answer
        content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
    if not content or not content.strip():
        raise AIError(f"{service} sent back an empty answer. Try again, or choose another model.")
    return content.strip()


# Turns the service's error code into what to do next: 401 means a wrong key, 429 the free
# limit, 500 and above means the service itself is overloaded.
def explain(service, status, body):
    if status in (401, 403) or "api key" in body.lower():
        return f"That {service} API key didn't work. Check it in the sidebar."
    if status == 429:
        return f"{service}'s free limit was reached. Wait a minute and try again, or switch to another service."
    if status >= 500:
        return f"{service} is busy right now. Try again in a minute, or choose another model or service."
    return f"{service} returned an error ({status}): {body[:300]}"


# Search only sees the question, not the chat. "And who coordinates the volunteers there?" has
# no clue what "there" is, so the AI first rewrites it as "Who coordinates the volunteers at the
# community garden?", and that's what gets searched. The first question needs no rewrite.
def standalone(service, model, key, history, question):
    """Turn a follow-up like "and when does it open?" into a full question, so the search finds the right passages."""
    if not history:
        return question
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY:])
    return ask(service, model, key, [{"role": "user", "content":
               f"{REWRITE_PROMPT}\n\nConversation:\n{conversation}\n\nLast question: {question}"}])


# Generation: the passages go into the instructions, each labeled with its file and place
# ("[budget.xlsx, sheet Budget, rows 2–5]"), then the recent conversation, then the question.
# The AI writes the answer from that, and the app shows the same labels as the sources.
def answer(service, model, key, history, question, passages):
    documents = "\n\n".join(f"[{p['file']}, {p['where']}]\n{p['text']}" for p in passages)
    messages = ([{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nDocuments:\n\n{documents}"}]
                + history[-HISTORY:] + [{"role": "user", "content": question}])
    return ask(service, model, key, messages)
