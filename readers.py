"""Turn uploaded files into passages of text, each with where it came from (page, sheet, slide).

Text formats are read inside the app. Scans, pictures and recordings are read by Gemini with the
user's own key, because they have no text to extract.
"""
import csv
import email
import email.policy
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import PurePosixPath
from xml.etree import ElementTree

ROWS_PER_PASSAGE = 40  # spreadsheet rows per passage; each passage repeats the header row

TEXT = {".txt", ".md", ".markdown", ".rst", ".log", ".tex", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".srt", ".vtt", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rb", ".php",
        ".r", ".sql", ".sh", ".css", ".swift", ".kt", ".rs", ".m", ".scala", ".jl"}
IMAGES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
          ".heic": "image/heic", ".heif": "image/heif", ".gif": "image/gif", ".bmp": "image/bmp",
          ".tif": "image/tiff", ".tiff": "image/tiff"}
AUDIO = {".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac",
         ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg", ".flac": "audio/flac",
         ".weba": "audio/webm"}
OTHER = {".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".pptx", ".odt", ".ods", ".odp",
         ".html", ".htm", ".xml", ".json", ".ipynb", ".rtf", ".epub", ".eml", ".msg", ".zip"}
SUPPORTED = sorted(TEXT | set(IMAGES) | set(AUDIO) | OTHER)
OLD_FORMATS = {".doc": "Word", ".ppt": "PowerPoint"}

READ_SCAN = ("Transcribe all text in this document exactly, in its original language. Keep tables as "
             "Markdown tables. Start each page with a line '=== Page N ===' where N is the page number. "
             "If a page has no text, describe what it shows in one or two sentences.")
READ_IMAGE = ("Transcribe all text in this image exactly, in its original language. Keep tables as Markdown "
              "tables. Then, after a line '---', describe what the image shows in a few sentences. "
              "If there is no text, give only the description.")
READ_AUDIO = ("Transcribe this recording word for word, in its original language. Start a new paragraph "
              "when the speaker changes, and label speakers as Speaker 1, Speaker 2 and so on.")


class Unreadable(Exception):
    """A file that can't be read, with a message the user can act on."""


def passage(text, file, where=""):
    return {"text": text.strip(), "file": file, "where": where}


def read(name, data, ask_ai):
    """[passage, ...] for one uploaded file. ask_ai(data, mime_type, prompt) -> text, using Gemini."""
    ext = PurePosixPath(name.lower()).suffix
    if ext in OLD_FORMATS:
        raise Unreadable(f"{name}: old {OLD_FORMATS[ext]} format. Open it and save it as {ext}x, then upload that.")
    reader = READERS.get(ext)
    if ext in TEXT:
        reader = read_text
    elif ext in IMAGES:
        return [passage(ask_ai(data, IMAGES[ext], READ_IMAGE), name, "picture")]
    elif ext in AUDIO:
        return [passage(ask_ai(data, AUDIO[ext], READ_AUDIO), name, "recording")]
    if reader is None:
        raise Unreadable(f"{name}: this kind of file isn't supported yet.")
    try:
        found = reader(name, data, ask_ai)
    except Unreadable:
        raise
    except Exception as error:  # a damaged or password-protected file
        raise Unreadable(f"{name}: couldn't be read ({type(error).__name__}). It may be damaged or password-protected.")
    found = [p for p in found if p["text"]]
    if not found:
        raise Unreadable(f"{name}: no text found in it.")
    return found


def decode(data):
    """Text from bytes: UTF-8, UTF-16 with its mark, or an old Windows code page (Cyrillic or Western)."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    high = [byte for byte in data if byte >= 0x80]
    cyrillic = sum(1 for byte in high if byte >= 0xC0)
    # Russian in cp1251 is almost all bytes 0xC0-0xFF; Western text has only a few accented letters
    russian = high and cyrillic / len(high) > 0.8 and len(high) > len(data) * 0.2
    return data.decode("cp1251" if russian else "cp1252", "replace")


def read_text(name, data, ask_ai):
    return [passage(decode(data), name)]


def read_pdf(name, data, ask_ai):
    from pypdf import PdfReader
    pdf = PdfReader(io.BytesIO(data))
    if pdf.is_encrypted:
        try:
            pdf.decrypt("")
        except Exception:
            raise Unreadable(f"{name}: the PDF is password-protected. Remove the password and upload it again.")
    pages = []
    for number, page in enumerate(pdf.pages, 1):
        label = pdf.page_labels[number - 1] if pdf.page_labels else str(number)
        try:
            pages.append((label, page.extract_text() or ""))
        except Exception:
            pages.append((label, ""))
    empty = sum(1 for _, text in pages if len(text.strip()) < 10)
    if pages and empty > len(pages) / 2:  # mostly scanned: let Gemini read the pages
        return split_pages(name, ask_ai(data, "application/pdf", READ_SCAN))
    return [passage(text, name, f"p. {label}") for label, text in pages]


def split_pages(name, text):
    parts = re.split(r"^=+\s*Page\s+(\d+)\s*=+\s*$", text, flags=re.M | re.I)
    if len(parts) < 3:
        return [passage(text, name, "scanned")]
    return [passage(parts[i + 1], name, f"p. {parts[i]}") for i in range(1, len(parts) - 1, 2)]


def read_docx(name, data, ask_ai):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    document = docx.Document(io.BytesIO(data))
    blocks, section = [], []
    heading = ""

    def flush():
        if section:
            blocks.append(passage("\n".join(section), name, heading))
            section.clear()

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            if paragraph.style is not None and paragraph.style.name.lower().startswith(("heading", "title")):
                flush()
                heading = text[:80]
            section.append(text)
        elif tag == "tbl":
            rows = [[cell.text.strip() for cell in row.cells] for row in Table(child, document).rows]
            section.append(markdown_table(rows))
    flush()
    for part in (document.sections[0].header, document.sections[0].footer) if document.sections else ():
        text = "\n".join(p.text for p in part.paragraphs if p.text.strip())
        if text:
            blocks.append(passage(text, name, "header and footer"))
    return blocks


def markdown_table(rows):
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [[cell.replace("\n", " ").replace("|", "/") for cell in row] + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def table_passages(name, sheet, rows):
    """Rows of one sheet as passages of ROWS_PER_PASSAGE rows, each with the header row repeated."""
    rows = [[("" if cell is None else str(cell)).strip() for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    header, body = rows[0], rows[1:] or [rows[0]]
    found = []
    for start in range(0, len(body), ROWS_PER_PASSAGE):
        chunk = body[start:start + ROWS_PER_PASSAGE]
        first, last = start + 2, start + 1 + len(chunk)
        where = (f"sheet {sheet}, " if sheet else "") + (f"rows {first}–{last}" if last > first else f"row {first}")
        found.append(passage(markdown_table([header] + chunk), name, where))
    return found


def read_xlsx(name, data, ask_ai):
    import openpyxl
    book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    found = []
    for sheet in book.worksheets:
        found += table_passages(name, sheet.title, sheet.iter_rows(values_only=True))
    return found


def read_xls(name, data, ask_ai):
    import xlrd
    book = xlrd.open_workbook(file_contents=data)
    found = []
    for sheet in book.sheets():
        found += table_passages(name, sheet.name, (sheet.row_values(i) for i in range(sheet.nrows)))
    return found


def read_csv(name, data, ask_ai):
    text = decode(data)
    try:
        dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if name.lower().endswith(".tsv") else csv.excel
    return table_passages(name, "", csv.reader(io.StringIO(text), dialect))


def read_pptx(name, data, ask_ai):
    from pptx import Presentation
    deck = Presentation(io.BytesIO(data))
    found = []
    for number, slide in enumerate(deck.slides, 1):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                lines += [p.text for p in shape.text_frame.paragraphs if p.text.strip()]
            if getattr(shape, "has_table", False) and shape.has_table:
                lines.append(markdown_table([[cell.text for cell in row.cells] for row in shape.table.rows]))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append("Speaker notes: " + notes)
        found.append(passage("\n".join(lines), name, f"slide {number}"))
    return found


ODF = "{urn:oasis:names:tc:opendocument:xmlns:%s:1.0}"
DRAW_PAGE = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}page"


def odf_text(element):
    """Text of an OpenDocument element, keeping spaces, tabs and line breaks."""
    parts = [element.text or ""]
    for child in element:
        tag = child.tag
        if tag == ODF % "text" + "s":
            parts.append(" " * int(child.get(ODF % "text" + "c", "1")))
        elif tag == ODF % "text" + "tab":
            parts.append("\t")
        elif tag == ODF % "text" + "line-break":
            parts.append("\n")
        else:
            parts.append(odf_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def read_odf(name, data, ask_ai):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ElementTree.fromstring(archive.read("content.xml"))
    body = root.find(ODF % "office" + "body")
    ext = PurePosixPath(name.lower()).suffix
    if ext == ".ods":
        found = []
        for table in body.iter(ODF % "table" + "table"):
            rows = []
            for row in table.iter(ODF % "table" + "table-row"):
                cells = []
                for cell in row.findall(ODF % "table" + "table-cell"):
                    repeat = min(int(cell.get(ODF % "table" + "number-columns-repeated", "1")), 50)
                    cells += ["\n".join(odf_text(p) for p in cell.iter(ODF % "text" + "p"))] * repeat
                repeat = min(int(row.get(ODF % "table" + "number-rows-repeated", "1")), 1000)
                rows += [cells] * repeat
            found += table_passages(name, table.get(ODF % "table" + "name", ""), rows)
        return found
    if ext == ".odp":
        return [passage("\n".join(odf_text(p) for p in page.iter(ODF % "text" + "p")), name, f"slide {number}")
                for number, page in enumerate(body.iter(DRAW_PAGE), 1)]
    paragraphs = [odf_text(p) for p in body.iter() if p.tag in (ODF % "text" + "p", ODF % "text" + "h")]
    return [passage("\n".join(p for p in paragraphs if p.strip()), name)]


class TextOnly(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skipping = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BREAK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, text):
        if not self.skipping:
            self.parts.append(text)


def html_text(html):
    parser = TextOnly()
    parser.feed(html)
    text = "".join(parser.parts)
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def read_html(name, data, ask_ai):
    return [passage(html_text(decode(data)), name)]


def read_xml(name, data, ask_ai):
    root = ElementTree.fromstring(data)
    return [passage("\n".join(t.strip() for t in root.itertext() if t.strip()), name)]


def read_json(name, data, ask_ai):
    value = json.loads(decode(data))
    return [passage(json.dumps(value, ensure_ascii=False, indent=1), name)]


def read_ipynb(name, data, ask_ai):
    notebook = json.loads(decode(data))
    found = []
    for number, cell in enumerate(notebook.get("cells", []), 1):
        source = "".join(cell.get("source", []))
        outputs = []
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/plain") or ""
            outputs.append("".join(text) if isinstance(text, list) else text)
        body = source + ("\n\nOutput:\n" + "\n".join(outputs) if any(outputs) else "")
        found.append(passage(body, name, f"cell {number}"))
    return found


def read_rtf(name, data, ask_ai):
    from striprtf.striprtf import rtf_to_text
    return [passage(rtf_to_text(decode(data), errors="ignore"), name)]


def read_epub(name, data, ask_ai):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        opf_path = next(e.get("full-path") for e in container.iter() if e.get("full-path"))
        opf = ElementTree.fromstring(archive.read(opf_path))
        base = PurePosixPath(opf_path).parent
        items = {e.get("id"): e.get("href") for e in opf.iter() if e.tag.endswith("item")}
        order = [e.get("idref") for e in opf.iter() if e.tag.endswith("itemref")]
        found = []
        for number, idref in enumerate(order, 1):
            href = items.get(idref)
            if not href:
                continue
            path = str(base / href) if str(base) != "." else href
            try:
                text = html_text(decode(archive.read(path)))
            except KeyError:
                continue
            first = next((line for line in text.splitlines() if line.strip()), "")[:60]
            found.append(passage(text, name, first or f"part {number}"))
        return found


def read_eml(name, data, ask_ai):
    message = email.message_from_bytes(data, policy=email.policy.default)
    head = "\n".join(f"{key}: {message[key]}" for key in ("From", "To", "Cc", "Date", "Subject") if message[key])
    body = message.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        text = body.get_content()
        if body.get_content_type() == "text/html":
            text = html_text(text)
    attachments = [part.get_filename() for part in message.iter_attachments() if part.get_filename()]
    found = [passage(head + "\n\n" + text, name, "email")]
    for part in message.iter_attachments():
        filename = part.get_filename()
        if filename and PurePosixPath(filename.lower()).suffix in set(SUPPORTED) - {".eml"}:
            try:
                found += [dict(p, file=f"{name} → {filename}") for p in read(filename, part.get_payload(decode=True), ask_ai)]
            except Unreadable:
                pass
    if attachments:
        found[0]["text"] += "\n\nAttachments: " + ", ".join(attachments)
    return found


def read_msg(name, data, ask_ai):
    import extract_msg
    message = extract_msg.openMsg(io.BytesIO(data))
    try:
        head = "\n".join(f"{key}: {value}" for key, value in
                         (("From", message.sender), ("To", message.to), ("Date", message.date), ("Subject", message.subject)) if value)
        return [passage(head + "\n\n" + (message.body or ""), name, "email")]
    finally:
        message.close()


def read_zip(name, data, ask_ai):
    found, total = [], 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            inner = info.filename
            if info.is_dir() or "__MACOSX" in inner or PurePosixPath(inner).name.startswith("."):
                continue
            total += info.file_size
            if total > 500_000_000:
                raise Unreadable(f"{name}: unpacks to more than 500 MB. Upload the files you need instead.")
            if PurePosixPath(inner.lower()).suffix not in SUPPORTED:
                continue
            try:
                found += [dict(p, file=f"{name} → {p['file']}") for p in read(inner, archive.read(info), ask_ai)]
            except Unreadable:
                continue
    return found


READERS = {".pdf": read_pdf, ".docx": read_docx, ".xlsx": read_xlsx, ".xlsm": read_xlsx, ".xls": read_xls,
           ".csv": read_csv, ".tsv": read_csv, ".pptx": read_pptx, ".odt": read_odf, ".ods": read_odf,
           ".odp": read_odf, ".html": read_html, ".htm": read_html, ".xml": read_xml, ".json": read_json,
           ".ipynb": read_ipynb, ".rtf": read_rtf, ".epub": read_epub, ".eml": read_eml, ".msg": read_msg,
           ".zip": read_zip}
