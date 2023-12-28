import fitz
from io import BytesIO

def pdf_to_string(file):
    # Gunakan PyMuPDF untuk membaca teks dari PDF
    pdf_document = fitz.open(file)
    text_content = ""

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text_content += page.get_text()

    pdf_document.close()

    return text_content

def request_pdf_to_string(file):
    try:
        # Membaca file PDF dari respons permintaan
        pdf_bytes = BytesIO(file.read())
        
        # Menggunakan PyMuPDF untuk membaca teks dari PDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_content = ""

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text_content += page.get_text()

        pdf_document.close()

        return text_content
    except Exception as e:
        return f'Error processing PDF: {str(e)}'