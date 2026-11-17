"""Ask the chosen AI service. Gemini, Mistral and Groq all accept the same OpenAI-style
request, so one function talks to all three."""
import requests

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
SYSTEM_PROMPT = (
    "You answer questions about the documents the user uploaded. Use only what the "
    "documents say. Get straight to the answer, without greetings. If the documents "
    "don't cover something, say so instead of guessing. Answer in the language of the question."
)
REWRITE_PROMPT = (
    "Rewrite the last question so it can be understood without the conversation before it. "
    "Keep its language. Reply with the rewritten question only."
)
HISTORY = 10  # earlier messages sent along, so follow-up questions make sense


class AIError(Exception):
    """The message is ready to show to the user."""


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


def explain(service, status, body):
    if status in (401, 403) or "api key" in body.lower():
        return f"That {service} API key didn't work. Check it in the sidebar."
    if status == 429:
        return f"{service}'s free limit was reached. Wait a minute and try again, or switch to another service."
    if status >= 500:
        return f"{service} is busy right now. Try again in a minute, or choose another model or service."
    return f"{service} returned an error ({status}): {body[:300]}"


def standalone(service, model, key, history, question):
    """Turn a follow-up like "and when does it open?" into a full question, so the search finds the right passages."""
    if not history:
        return question
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY:])
    return ask(service, model, key, [{"role": "user", "content":
               f"{REWRITE_PROMPT}\n\nConversation:\n{conversation}\n\nLast question: {question}"}])


def answer(service, model, key, history, question, passages):
    documents = "\n\n".join(f"[{p['file']}, {p['where']}]\n{p['text']}" for p in passages)
    messages = ([{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nDocuments:\n\n{documents}"}]
                + history[-HISTORY:] + [{"role": "user", "content": question}])
    return ask(service, model, key, messages)
