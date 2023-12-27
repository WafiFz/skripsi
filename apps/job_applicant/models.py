from apps import db
from datetime import datetime

class JobApplicants(db.Model):

    __tablename__ = 'job_applicants'

    id = db.Column(db.Integer, primary_key=True)
    cv_path = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('Users.id'), nullable=False)
    desired_job = db.Column(db.String(255), nullable=True)

    # Define the relationship with the Users table
    user = db.relationship('Users', back_populates='job_applications')

    def __repr__(self):
        return f"Job Applicant(id={self.id}, user_id={self.user_id}, desired_job={self.desired_job}, upload_date={self.upload_date})"