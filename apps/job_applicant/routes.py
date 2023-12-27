from apps.job_applicant import blueprint

@blueprint.route('/')
def index():
    return "OK"