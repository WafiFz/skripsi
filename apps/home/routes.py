from apps.home import blueprint
from flask import jsonify, render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import fitz
from io import BytesIO
import torch
from transformers import BertTokenizer
from .BERTClass import predict_category
from .IndoBERTClass import indo_predict_category
from .DataPreprocessing import text_preprocessing

@blueprint.route('/')
def index():

    return render_template('home/index.html', segment='index')

@blueprint.route('/', methods=['POST'])
def index_post():
    try:
        file = request.files['file']
        is_indonesia = request.form['is_indonesia'] == 'true'

        # Check if the file is present
        if 'file' not in request.files or file.filename == '':
            raise ValueError('No file uploaded.')

        # Check if the file has a PDF extension
        if not file.filename.lower().endswith('.pdf'):
            raise ValueError('Invalid file format. Please upload a PDF file.')
        
        # Read the PDF file and convert it to a string
        text = request_pdf_to_string(file)

        # Check if the file content is empty
        if not text.strip():
            raise ValueError('Empty file content. Please upload a non-empty PDF file.')

        text = text_preprocessing(text, is_indonesia)
        # return jsonify({'success': True, 'output': 'test_output', 'probability': 0.7877, 'text' : text})

        # Predict
        success, output, probability =  ( indo_predict_category(text) if is_indonesia 
                                          else predict_category(text)
                                        )
        
        if not success:
            raise ValueError(output)
        
        return jsonify({'success': True, 'output': output, 'probability': probability, 'text' : text})
        # return jsonify({'success': True, 'output': f'Hasil:  {str(output)}\n Probabilitas:  {str(probability)}'})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)}), 500

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