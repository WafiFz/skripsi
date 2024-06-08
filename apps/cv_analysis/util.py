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
    
def translate_label_to_indonesian(english_label):
    translation_dict = {
        'Accountant': 'Akuntan', 
        'Advocate': 'Advokat', 
        'Agriculture': 'Pertanian', 
        'Apparel': 'Pakaian', 
        'Arts': 'Seni', 
        'Automobile': 'Otomotif', 
        'Aviation': 'Penerbangan', 
        'Banking': 'Perbankan', 
        'BPO (Business Process Outsourcing)': 'BPO (Bisnis Proses Outsourcing)', 
        'Business-development': 'Pengembangan Bisnis', 
        'Chef': 'Koki', 
        'Construction': 'Konstruksi', 
        'Consultant': 'Konsultan', 
        'Designer': 'Desainer', 
        'Digital-media': 'Media Digital', 
        'Engineering': 'Engineering',
        'Finance': 'Keuangan', 
        'Fitness': 'Kebugaran',
        'Healthcare': 'Perawatan Kesehatan', 
        'Hr': 'SDM (Sumber Daya Manusia)', 
        'Information-technology': 'Teknologi Informasi', 
        'Public-relations': 'Hubungan Masyarakat', 
        'Sales': 'Sales',
        'Teacher': 'Guru'
    }
    
    return translation_dict.get(english_label, 'Label tidak ditemukan')