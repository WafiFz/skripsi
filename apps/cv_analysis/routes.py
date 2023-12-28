from flask import jsonify, render_template, request
from apps.cv_analysis import blueprint
from apps.cv_analysis.models import CvAnalysisResults
from apps.cv_analysis.util import request_pdf_to_string
from apps.cv_analysis.DataPreprocessing import text_preprocessing
from apps.cv_analysis.BERTClass import predict_category
from apps.cv_analysis.IndoBERTClass import indo_predict_category
from apps.cv_analysis.DataPreprocessing import text_preprocessing

@blueprint.route('/quick-cv-prediction')
def quick_cv_prediction():
    return render_template('cv_analysis/quick-cv-prediction.html')

@blueprint.route('/quick-cv-prediction', methods=['POST'])
def quick_cv_prediction_post():
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
        
        # return jsonify({'success': True, 'output': output, 'probability': probability, 'text' : text})
        return jsonify({
            'success': True,
            'output': f'Result: {str(output)}<br/> Probability: {probability:.2%}'
        })
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)}), 500