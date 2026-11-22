# Ask your data

Upload your files and ask questions about them. Every answer lists where it came from: the page, the sheet and rows, or the slide.

**Try it:** https://ask-your-db.streamlit.app

## What it reads

| Kind | Files | Sources point to |
|---|---|---|
| Documents | Word (.docx), OpenDocument (.odt), PDF, RTF, plain text, Markdown | page or section |
| Spreadsheets | Excel (.xlsx, .xlsm, .xls), OpenDocument (.ods), CSV, TSV | sheet and rows |
| Slides | PowerPoint (.pptx, with speaker notes), OpenDocument (.odp) | slide |
| Email and books | .eml, Outlook .msg (with attachments), EPUB | email or chapter |
| Web and data | HTML, JSON, XML, YAML, Jupyter notebooks, code files | the file |
| Archives | .zip: the files inside are read | the file inside |

## How it works

1. You choose an AI service (Gemini, Mistral or Groq) and paste your own free API key in the sidebar. It stays in your browser session and isn't stored.
2. The app splits your files into passages, keeping where each one came from, and indexes them with a small free model that runs inside the app, so searching costs nothing.
3. For each question, it searches two ways at once: by meaning, and by the exact words, which catches names, codes and numbers. The 8 best passages (5 for Groq, whose free limit is smaller) go to the AI service you chose, which writes the answer from them and remembers the conversation.

Answers come in the language of the question, and the app says so when the documents don't cover something.

The code is four short files and no AI framework: `readers.py` turns each file type into text, `search.py` cuts it into passages and finds the best ones, `ai.py` sends them to the service you chose, and `app.py` is the screen.

## Deploy your own

1. Click **Use this template** at the top of this page and create your own copy of the repo.
2. Go to https://share.streamlit.io, click **Create app**, and choose your copy, branch `main`, file `app.py`.
3. Under **Advanced settings**, set Python to **3.12**. No secrets are needed.
4. Click **Deploy**, then paste a free API key in the app's sidebar: [Gemini](https://aistudio.google.com/apikey), [Mistral](https://console.mistral.ai/api-keys) or [Groq](https://console.groq.com/keys).

To try it without installing anything, open your copy in GitHub Codespaces (**Code**, then **Codespaces**). It installs everything and starts the app.

## Run it locally

1. Get a free API key from [Gemini](https://aistudio.google.com/apikey), [Mistral](https://console.mistral.ai/api-keys) or [Groq](https://console.groq.com/keys).
2. Install the requirements (Python 3.12):
   ```
   pip install -r requirements.txt
   ```
3. Start the app and paste your key in the sidebar:
   ```
   streamlit run app.py
   ```

## Tests

Each format has a test that makes a small made-up file and reads it back. The search and the AI calls are tested with stand-ins, so no key or internet is needed:
```
python -m unittest discover -s tests
```

## Tools

Python, Streamlit, FastEmbed, NumPy, BM25, Google Gemini, Mistral, Groq, pypdf, python-docx, openpyxl, python-pptx

## License

MIT. See [LICENSE](LICENSE).

The icons are from Google [Material Symbols](https://fonts.google.com/icons), under the Apache License 2.0.
