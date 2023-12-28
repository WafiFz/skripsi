from flask import jsonify, render_template, request
from flask_login import login_required
from apps import db
from apps.home import blueprint
from apps.job_applicant.models import JobApplicants
from apps.cv_analysis.util import pdf_to_string
from apps.cv_analysis.BERTClass import predict_category
from apps.cv_analysis.IndoBERTClass import indo_predict_category
from apps.cv_analysis.DataPreprocessing import text_preprocessing

@blueprint.route('/')
def index():
    return render_template('home/index.html', segment='index')

@blueprint.route('/admin/prediction-result')
def prediction_result():
    return render_template('admin/prediction-result.html')

@blueprint.route('/admin/mapping-position')
def mapping_position():
    return render_template('admin/mapping-position.html')

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
        
        # Save PDF
        file_path = 'cv_users/' + file.filename
        file.save(file_path)

        # Create Job Applicant entry
        user_id =  2# retrieve user_id from your authentication mechanism
        desired_job = request.form.get('desired_job', None)  # Assuming you have a field 'desired_job' in your form
        
        job_applicant = JobApplicants(
            cv_path=file_path,
            user_id=user_id,
            desired_job=desired_job
        )
        
        db.session.add(job_applicant)
        db.session.commit()
        
        # Read the PDF file and convert it to a string
        text = pdf_to_string(file_path)

        # Check if the file content is empty
        if not text.strip():
            raise ValueError('Empty file content. Please upload a non-empty PDF file.')

        text = text_preprocessing(text, is_indonesia)
        # return jsonify({'success': True, 'output': 'test_output', 'probability': 0.0, 'text' : text})

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