"""Step 1 of RAG (retrieval-augmented generation): turn uploaded files into plain text.

RAG answers a question in three steps:
  1. read: every file becomes passages of text, each labeled with where it came from (this file)
  2. retrieve: the passages that best match the question are found (search.py)
  3. generate: an AI writes the answer from those passages only (ai.py)

The AI never sees your files, only the text this file pulls out of them. So the labels made
here (page, sheet and rows, slide) are what the answer's sources point to.
Everything is read inside the app; nothing is sent anywhere at this step."""
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
OTHER = {".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".pptx", ".odt", ".ods", ".odp",
         ".html", ".htm", ".xml", ".json", ".ipynb", ".rtf", ".epub", ".eml", ".msg", ".zip"}
SUPPORTED = sorted(TEXT | OTHER)
OLD_FORMATS = {".doc": "Word", ".ppt": "PowerPoint"}


# When a file can't become text, the user gets a plain reason instead of a silent gap in the answers.
class Unreadable(Exception):
    """A file that can't be read, with a message the user can act on."""


# The unit everything else works with: a piece of text plus its source.
# search.py cuts long ones smaller, and the file and where travel with every piece to the answer.
def passage(text, file, where=""):
    return {"text": text.strip(), "file": file, "where": where}


# The entry point: picks the right reader by file extension, and turns damaged, old or empty
# files into clear messages. Every reader below returns a list of passages.
def read(name, data):
    """[passage, ...] for one uploaded file."""
    ext = PurePosixPath(name.lower()).suffix
    if ext in OLD_FORMATS:
        raise Unreadable(f"{name}: old {OLD_FORMATS[ext]} format. Open it and save it as {ext}x, then upload that.")
    reader = READERS.get(ext)
    if ext in TEXT:
        reader = read_text
    if reader is None:
        raise Unreadable(f"{name}: this kind of file isn't supported yet.")
    try:
        found = reader(name, data)
    except Unreadable:
        raise
    except Exception as error:  # a damaged or password-protected file
        raise Unreadable(f"{name}: couldn't be read ({type(error).__name__}). It may be damaged or password-protected.")
    found = [p for p in found if p["text"]]
    if not found:
        raise Unreadable(f"{name}: no text found in it.")
    return found


# Plain-text files carry no label for their encoding. Guessing wrong turns Russian into
# gibberish, and gibberish can't be found by search, so this checks the bytes first.
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


# Text, Markdown, code and similar files: already text, so one passage for the whole file.
def read_text(name, data):
    return [passage(decode(data), name)]


# One passage per page, so a source can say "p. 12". A scanned PDF is only pictures of pages
# with no text inside, so it's flagged instead of silently giving nothing.
def read_pdf(name, data):
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
    if pages and empty > len(pages) / 2:
        raise Unreadable(f"{name}: looks like a scan, so there's no text to read. Run it through OCR first "
                         "(for example, Adobe Acrobat or Google Drive's 'Open with Google Docs').")
    return [passage(text, name, f"p. {label}") for label, text in pages]


# One passage per section, cut at each heading, so a source can name the section.
# Tables become Markdown tables, which keeps each value next to its column name.
def read_docx(name, data):
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


# Rows and columns as text the AI can read: | a | b |. Without the layout,
# "980" would lose the "Fence" and "Cost" that give it meaning.
def markdown_table(rows):
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [[cell.replace("\n", " ").replace("|", "/") for cell in row] + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


# Spreadsheets are cut into blocks of rows, and each block repeats the header row,
# so row 412 still says which column is "Date" and which is "Amount".
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


# Excel: every sheet, read as values (not formulas), cut into row blocks.
def read_xlsx(name, data):
    import openpyxl
    book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    found = []
    for sheet in book.worksheets:
        found += table_passages(name, sheet.title, sheet.iter_rows(values_only=True))
    return found


# The older Excel format (.xls), same treatment.
def read_xls(name, data):
    import xlrd
    book = xlrd.open_workbook(file_contents=data)
    found = []
    for sheet in book.sheets():
        found += table_passages(name, sheet.name, (sheet.row_values(i) for i in range(sheet.nrows)))
    return found


# CSV and TSV: the separator (comma, semicolon, tab) is detected, then cut like a spreadsheet.
def read_csv(name, data):
    text = decode(data)
    try:
        dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel_tab if name.lower().endswith(".tsv") else csv.excel
    return table_passages(name, "", csv.reader(io.StringIO(text), dialect))


# One passage per slide, including tables and speaker notes, so a source can say "slide 4".
def read_pptx(name, data):
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


# OpenDocument (LibreOffice) files are zipped XML. This collects the text of one element,
# turning the format's space, tab and line-break codes back into real ones.
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


# LibreOffice and Google Docs exports: .odt as one passage, .ods sheets like a spreadsheet,
# .odp one passage per slide.
def read_odf(name, data):
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


# Web pages: keeps the visible text and drops scripts, styles and page code, which would
# only add noise to the search.
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


# Runs the parser above and tidies the spacing.
def html_text(html):
    parser = TextOnly()
    parser.feed(html)
    text = "".join(parser.parts)
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


# Saved web pages: their visible text as one passage.
def read_html(name, data):
    return [passage(html_text(decode(data)), name)]


# XML: just the text between the tags.
def read_xml(name, data):
    root = ElementTree.fromstring(data)
    return [passage("\n".join(t.strip() for t in root.itertext() if t.strip()), name)]


# JSON: laid out with one value per line, so names and values stay readable together.
def read_json(name, data):
    value = json.loads(decode(data))
    return [passage(json.dumps(value, ensure_ascii=False, indent=1), name)]


# Jupyter notebooks: one passage per cell, with its code and its output.
def read_ipynb(name, data):
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


# RTF documents: formatting codes removed, text kept.
def read_rtf(name, data):
    from striprtf.striprtf import rtf_to_text
    return [passage(rtf_to_text(decode(data), errors="ignore"), name)]


# E-books: one passage per chapter, in reading order, named by the chapter's first line.
def read_epub(name, data):
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


# Emails (.eml): sender, date and subject stay with the body, so you can ask who wrote what
# and when. Attachments are read too, and their sources say which email they came from.
def read_eml(name, data):
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
                found += [dict(p, file=f"{name} → {filename}") for p in read(filename, part.get_payload(decode=True))]
            except Unreadable:
                pass
    if attachments:
        found[0]["text"] += "\n\nAttachments: " + ", ".join(attachments)
    return found


# Outlook emails (.msg): sender, date, subject and body.
def read_msg(name, data):
    import extract_msg
    message = extract_msg.openMsg(io.BytesIO(data))
    try:
        head = "\n".join(f"{key}: {value}" for key, value in
                         (("From", message.sender), ("To", message.to), ("Date", message.date), ("Subject", message.subject)) if value)
        return [passage(head + "\n\n" + (message.body or ""), name, "email")]
    finally:
        message.close()


# Zip files: every supported file inside is read, and its source shows the path inside the zip.
# Stops at 500 MB unpacked, so one upload can't fill the app's memory.
def read_zip(name, data):
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
                found += [dict(p, file=f"{name} → {p['file']}") for p in read(inner, archive.read(info))]
            except Unreadable:
                continue
    return found


# Which reader handles which extension.
READERS = {".pdf": read_pdf, ".docx": read_docx, ".xlsx": read_xlsx, ".xlsm": read_xlsx, ".xls": read_xls,
           ".csv": read_csv, ".tsv": read_csv, ".pptx": read_pptx, ".odt": read_odf, ".ods": read_odf,
           ".odp": read_odf, ".html": read_html, ".htm": read_html, ".xml": read_xml, ".json": read_json,
           ".ipynb": read_ipynb, ".rtf": read_rtf, ".epub": read_epub, ".eml": read_eml, ".msg": read_msg,
           ".zip": read_zip}
