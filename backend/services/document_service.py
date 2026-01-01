from io import BytesIO
from pypdf import PdfReader
import docx

def extract_text_from_file(content: bytes, ext: str) -> str:
    """
    Verilen dosya içeriğinden (bytes) metin ayıklar.
    Desteklenen uzantılar: pdf, docx.
    """
    ext = ext.lower()
    if ext == "pdf":
        return extract_text_from_pdf(content)
    elif ext == "docx":
        return extract_text_from_docx(content)
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")

def extract_text_from_pdf(content: bytes) -> str:
    text = ""
    reader = PdfReader(BytesIO(content))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(content: bytes) -> str:
    file_stream = BytesIO(content)
    doc = docx.Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])