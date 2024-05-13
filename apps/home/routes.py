from flask import jsonify, render_template, request
from flask_login import login_required, current_user
from apps import db
from apps.home import blueprint
from apps.authentication.routes import role_admin
from apps.job_applicant.models import JobApplicants
from apps.cv_analysis.models import CvAnalysisResults
from apps.cv_analysis.util import pdf_to_string
from apps.cv_analysis.bert.english_cv_predict import english_predict_category
from apps.cv_analysis.bert.indonesian_cv_predict import indo_predict_category
from apps.cv_analysis.DataPreprocessing import text_preprocessing
import os

@blueprint.route('/')
@login_required
def index():
    return render_template('home/index.html', segment='index')

@blueprint.route('/admin')
@login_required
def dashboard():
    is_admin = role_admin(current_user.role)
    if not is_admin:
        return render_template('home/index.html', segment='index')
    
    prediction_results = CvAnalysisResults.query.all()

    return render_template('admin/index.html', prediction_results=prediction_results)

@blueprint.route('/admin/prediction-result')
@login_required
def prediction_result():
    is_admin = role_admin(current_user.role)
    if not is_admin:
        return render_template('home/index.html', segment='index')
    
    prediction_results = CvAnalysisResults.query.all()

    return render_template('admin/prediction-result.html', prediction_results=prediction_results)

@blueprint.route('/admin/mapping-position')
@login_required
def mapping_position():
    is_admin = role_admin(current_user.role)
    if not is_admin:
        return render_template('home/index.html', segment='index')
    
    prediction_results = CvAnalysisResults.query.all()

    return render_template('admin/mapping-position.html', prediction_results=prediction_results)

@blueprint.route('/admin/cv-preview/<int:job_applicant_id>')
@login_required
def cv_preview(job_applicant_id):
    is_admin = role_admin(current_user.role)
    if not is_admin:
        return render_template('home/index.html', segment='index')
    
    job_applicant = JobApplicants.query.get(job_applicant_id)  

    return render_template('admin/cv-preview.html', job_applicant=job_applicant)

@blueprint.route('/', methods=['POST'])
@login_required
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

        save_folder='apps/static/cv_users'

        # Ensure directory exists or create if not
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Save PDF
        file_path = os.path.join(save_folder, file.filename)
        file.save(file_path)

        # Create Job Applicant entry
        user_id = current_user.id
        desired_job = request.form.get('desired_job', None)

        # Check if there is an existing JobApplicant entry for the same user_id
        existing_job_applicant = JobApplicants.query.filter_by(user_id=user_id).first()

        if existing_job_applicant:
            raise ValueError('You have already uploaded your CV.')

        job_applicant = JobApplicants(
            cv_path=file.filename,
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
        
        # Predict
        success, output, probability = (indo_predict_category(text) if is_indonesia 
                                        else english_predict_category(text))

        # Save prediction results to CvAnalysisResults table
        status = 'pass' if output == desired_job else 'fail'

        # Round the probability value to four decimal places
        rounded_probability = round(probability, 4)

        cv_analysis_result = CvAnalysisResults(
            job_applicant_id=job_applicant.id,
            prediction_result=output,
            probability_result=rounded_probability,
            status=status
        )

        db.session.add(cv_analysis_result)
        db.session.commit()

        if not success:
            raise ValueError(output)
        
        return jsonify({'success': True, 'output': 'Your CV has been uploaded. Contact the admin at 089656377911 if you want to know the result.', 'probability': probability})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)}), 500