import torch
import numpy as np
from apps.cv_analysis.bert.BERTClass import BERTClass
from apps.cv_analysis.bert.util import (
    download_model, 
    define_device,
    define_tokenizer, 
    predict_category, 
)

class_labels = [
    'Accountant', 
    'Advocate', 
    'Agriculture', 
    'Apparel', 
    'Arts', 
    'Automobile', 
    'Aviation', 
    'Banking', 
    'BPO (Business Process Outsourcing)', 
    'Business-development', 
    'Chef', 
    'Construction', 
    'Consultant', 
    'Designer', 
    'Digital-media', 
    'Engineering', 
    'Finance', 
    'Fitness',
    'Healthcare', 
    'Hr', 
    'Information-technology', 
    'Public-relations', 
    'Sales', 
    'Teacher'
]

BERT = 'bert-base-uncased'
MODEL_PRE_TRAINED = BERT

MODEL_URL = 'https://cloud.wafi.web.id/cv_model_v5.pth'
MODEL_FOLDER = 'trained_model_online'
MODEL_FILE = 'cv_model_v5.pth'

# Set the model path
model_path = download_model(MODEL_FOLDER, MODEL_FILE, MODEL_URL)

# Check is CUDA available
device = define_device()

# Load the trained model
model = BERTClass(MODEL_PRE_TRAINED)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the tokenizer
tokenizer = define_tokenizer(MODEL_PRE_TRAINED)

def english_predict_category(resume):
    is_success = False
    try:
        return predict_category(resume, tokenizer, model, class_labels)

    except Exception as e:
        return is_success, f'Error in prediction:  {str(e)}', 0