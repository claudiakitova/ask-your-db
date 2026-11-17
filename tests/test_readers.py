"""Each format is made here in a tiny made-up file and read back. Run: python -m unittest discover -s tests"""
import io
import sys
import unittest
import zipfile
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import readers  # noqa: E402

def read(name, data):
    return readers.read(name, data)


def text_of(parts):
    return "\n".join(p["text"] for p in parts)


def tiny_pdf(pages):
    """A valid PDF with one line of text per page ("" for a page with no text)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for number, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return out.getvalue()


def zipped(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return out.getvalue()


ODF_HEAD = ('<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
            'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
            'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"><office:body>')


class Formats(unittest.TestCase):
    def test_pdf_pages(self):
        parts = read("report.pdf", tiny_pdf(["Revenue grew 12 percent", "Costs fell"]))
        self.assertEqual([p["where"] for p in parts], ["p. 1", "p. 2"])
        self.assertIn("Revenue grew 12 percent", parts[0]["text"])

    def test_scanned_pdf_explains(self):
        with self.assertRaisesRegex(readers.Unreadable, "looks like a scan"):
            read("scan.pdf", tiny_pdf(["", ""]))

    def test_docx_with_headings_and_table(self):
        import docx
        document = docx.Document()
        document.add_heading("Budget", 1)
        document.add_paragraph("We plan to spend less on travel.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text, table.cell(0, 1).text = "Item", "Cost"
        table.cell(1, 0).text, table.cell(1, 1).text = "Flights", "900"
        document.add_heading("Team", 1)
        document.add_paragraph("Anna leads the project.")
        out = io.BytesIO(); document.save(out)
        parts = read("plan.docx", out.getvalue())
        self.assertEqual([p["where"] for p in parts], ["Budget", "Team"])
        self.assertIn("| Flights | 900 |", parts[0]["text"])

    def test_xlsx_rows_keep_their_header(self):
        import openpyxl
        book = openpyxl.Workbook(); sheet = book.active; sheet.title = "Sales"
        sheet.append(["Month", "Units"])
        for month in range(1, 51):
            sheet.append([f"M{month}", month * 10])
        out = io.BytesIO(); book.save(out)
        parts = read("sales.xlsx", out.getvalue())
        self.assertEqual([p["where"] for p in parts], ["sheet Sales, rows 2–41", "sheet Sales, rows 42–51"])
        self.assertTrue(parts[1]["text"].startswith("| Month | Units |"))
        self.assertIn("| M50 | 500 |", parts[1]["text"])

    def test_csv_with_semicolons(self):
        parts = read("people.csv", "name;city\nAnna;Berlin\nLeo;Lisbon\n".encode())
        self.assertIn("| Leo | Lisbon |", parts[0]["text"])

    def test_pptx_slides_and_notes(self):
        from pptx import Presentation
        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = "Launch plan"
        slide.notes_slide.notes_text_frame.text = "Mention the March date."
        out = io.BytesIO(); deck.save(out)
        parts = read("deck.pptx", out.getvalue())
        self.assertEqual(parts[0]["where"], "slide 1")
        self.assertIn("Speaker notes: Mention the March date.", parts[0]["text"])

    def test_odt_ods_odp(self):
        odt = zipped({"content.xml": ODF_HEAD + "<office:text><text:h>Notes</text:h><text:p>Hello<text:s/>world</text:p>"
                      "</office:text></office:body></office:document-content>"})
        self.assertIn("Hello world", text_of(read("notes.odt", odt)))
        ods = zipped({"content.xml": ODF_HEAD + '<office:spreadsheet><table:table table:name="Q1">'
                      "<table:table-row><table:table-cell><text:p>City</text:p></table:table-cell></table:table-row>"
                      "<table:table-row><table:table-cell><text:p>Porto</text:p></table:table-cell></table:table-row>"
                      "</table:table></office:spreadsheet></office:body></office:document-content>"})
        parts = read("cities.ods", ods)
        self.assertEqual(parts[0]["where"], "sheet Q1, row 2")
        odp = zipped({"content.xml": ODF_HEAD + "<office:presentation><draw:page><text:p>Slide one</text:p></draw:page>"
                      "</office:presentation></office:body></office:document-content>"})
        self.assertEqual(read("talk.odp", odp)[0]["where"], "slide 1")

    def test_html_skips_scripts(self):
        html = b"<html><head><title>x</title><script>var a=1</script></head><body><h1>Prices</h1><p>Tea 3</p></body></html>"
        text = text_of(read("page.html", html))
        self.assertIn("Prices", text); self.assertNotIn("var a", text)

    def test_text_json_xml_rtf_notebook(self):
        self.assertIn("Привет", text_of(read("note.txt", "Привет".encode("cp1251"))))
        self.assertIn('"city": "Oslo"', text_of(read("data.json", b'{"city": "Oslo"}')))
        self.assertIn("Oslo", text_of(read("data.xml", b"<a><city>Oslo</city></a>")))
        self.assertIn("Bold words", text_of(read("letter.rtf", rb"{\rtf1\ansi {\b Bold words}}")))
        notebook = b'{"cells": [{"cell_type": "code", "source": ["print(2+2)"], "outputs": [{"text": ["4"]}]}]}'
        parts = read("analysis.ipynb", notebook)
        self.assertEqual(parts[0]["where"], "cell 1"); self.assertIn("Output:\n4", parts[0]["text"])

    def test_epub_chapters(self):
        epub = zipped({
            "META-INF/container.xml": '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>',
            "OEBPS/book.opf": '<package><manifest><item id="c1" href="one.xhtml"/></manifest><spine><itemref idref="c1"/></spine></package>',
            "OEBPS/one.xhtml": "<html><body><h1>Chapter One</h1><p>It was raining.</p></body></html>"})
        parts = read("book.epub", epub)
        self.assertEqual(parts[0]["where"], "Chapter One"); self.assertIn("It was raining.", parts[0]["text"])

    def test_email_with_attachment(self):
        message = EmailMessage()
        message["From"], message["To"], message["Subject"] = "anna@example.com", "me@example.com", "Contract"
        message.set_content("See the attached terms.")
        message.add_attachment(b"Payment within 30 days.", maintype="text", subtype="plain", filename="terms.txt")
        parts = read("mail.eml", message.as_bytes())
        self.assertIn("Subject: Contract", parts[0]["text"])
        self.assertIn("Attachments: terms.txt", parts[0]["text"])
        self.assertEqual(parts[1]["file"], "mail.eml → terms.txt")

    def test_zip_reads_inside(self):
        archive = zipped({"a/notes.md": "# Trip\nWe went to Kyoto.", "__MACOSX/._x": "junk", "b/photo.jpg": b"\xff\xd8"})
        parts = read("bundle.zip", archive)
        self.assertEqual({p["file"] for p in parts}, {"bundle.zip → a/notes.md"})  # the photo is skipped

    def test_old_formats_and_damaged_files_explain(self):
        with self.assertRaisesRegex(readers.Unreadable, "save it as .docx"):
            read("old.doc", b"x")
        with self.assertRaisesRegex(readers.Unreadable, "damaged or password-protected"):
            read("broken.xlsx", b"not a spreadsheet")
        with self.assertRaisesRegex(readers.Unreadable, "no text"):
            read("empty.txt", b"   ")


if __name__ == "__main__":
    unittest.main()
