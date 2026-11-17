# Chat with your data

Upload your files and ask questions about them. Every answer lists where it came from: the page, the sheet and rows, or the slide.

**Try it:** https://ask-your-db.streamlit.app

## What it reads

| Kind | Files | Sources point to |
|---|---|---|
| Documents | Word (.docx), OpenDocument (.odt), PDF, RTF, plain text, Markdown | page or section |
| Spreadsheets | Excel (.xlsx, .xlsm, .xls), OpenDocument (.ods), CSV, TSV | sheet and rows |
| Slides | PowerPoint (.pptx, with speaker notes), OpenDocument (.odp) | slide |
| Scans and pictures | Scanned PDFs, photos, screenshots (.png, .jpg, .heic and others) | page |
| Recordings | .mp3, .m4a, .wav, .ogg, .flac | the recording |
| Email and books | .eml, Outlook .msg (with attachments), EPUB | email or chapter |
| Web and data | HTML, JSON, XML, YAML, Jupyter notebooks, code files | the file |
| Archives | .zip: the files inside are read | the file inside |

Text files are read inside the app. Scans, pictures and recordings have no text to pull out, so Gemini reads them with your key: it writes out the text, describes pictures, and transcribes recordings. Old .doc and .ppt files need to be saved as .docx or .pptx first.

## How it works

1. You paste your own free Gemini API key in the sidebar. It stays in your browser session and isn't stored.
2. The app splits your files into passages, keeping where each one came from, and indexes them with a small free model that runs inside the app, so searching costs nothing.
3. For each question, it finds the most relevant passages, and Gemini writes the answer from them, remembering the conversation.

Answers come in the language of the question, and the app says so when the documents don't cover something.

## Deploy your own

1. Click **Use this template** at the top of this page and create your own copy of the repo.
2. Go to https://share.streamlit.io, click **Create app**, and choose your copy, branch `main`, file `app.py`.
3. Under **Advanced settings**, set Python to **3.12**. No secrets are needed.
4. Click **Deploy**, then paste your free [Gemini API key](https://aistudio.google.com/apikey) in the app's sidebar.

To try it without installing anything, open your copy in GitHub Codespaces (**Code**, then **Codespaces**). It installs everything and starts the app.

## Run it locally

1. Get a free Gemini API key at https://aistudio.google.com/apikey.
2. Install the requirements (Python 3.12):
   ```
   pip install -r requirements.txt
   ```
3. Start the app and paste your key in the sidebar:
   ```
   streamlit run app.py
   ```

## Tests

Each format has a test that makes a small made-up file and reads it back:
```
python -m unittest discover -s tests
```

## Tools

Python, Streamlit, LlamaIndex, Google Gemini, FastEmbed, pypdf, python-docx, openpyxl, python-pptx

## License

MIT. See [LICENSE](LICENSE).
