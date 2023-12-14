from apps.cv_analysis import blueprint

@blueprint.route('/')
def index():
    return "OK"