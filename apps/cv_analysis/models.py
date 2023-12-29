from apps import db
from datetime import datetime
from apps.job_applicant.models import JobApplicants

class CvAnalysisResults(db.Model):

    __tablename__ = 'Cv_analysis_results'

    id = db.Column(db.Integer, primary_key=True)
    job_applicant_id = db.Column(db.Integer, db.ForeignKey('Job_applicants.id'), nullable=False)
    prediction_result = db.Column(db.String(255), nullable=False)
    probability_result = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(255), nullable=False)
    analysis_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Define the relationship with the JobAplicants table
    job_applicant = db.relationship('JobApplicants', back_populates='cv_analysis_result')

    def __repr__(self):
        return f"CvAnalysisResults(id={self.id}, job_applicant_id={self.job_applicant_id}, prediction_result={self.prediction_result}, analysis_at={self.analysis_at})"