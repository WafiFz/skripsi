# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import jsonify, render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import fitz
from io import BytesIO


@blueprint.route('/index')
def index():

    return render_template('home/index.html', segment='index')

@blueprint.route('/index', methods=['POST'])
def index_post():
    file = request.files['file']

    # Check if the file has a PDF extension
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'output': 'Invalid file format. Please upload a PDF file.'}), 400

    # Read the PDF file and convert it to a string
    try:
        text = request_pdf_to_string(file)

        # Lakukan sesuatu dengan text_content, misalnya, print atau kirim sebagai tanggapan
        print(text)

        return jsonify({'success': True, 'output': text}), 400
    except Exception as e:
        return jsonify({'success': False, 'output': f'Error processing PDF: {str(e)}'}), 500

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