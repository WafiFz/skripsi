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
    'Akuntan', 
    'Advokat', 
    'Pertanian', 
    'Pakaian', 
    'Seni', 
    'Otomotif', 
    'Penerbangan', 
    'Perbankan', 
    'BPO (Bisnis Proses Outsourcing)', 
    'Pengembangan Bisnis', 
    'Koki', 
    'Konstruksi', 
    'Konsultan', 
    'Desainer', 
    'Media Digital', 
    'Engineering', 
    'Keuangan', 
    'Kebugaran', 
    'Perawatan Kesehatan', 
    'SDM (Sumber Daya Manusia)', 
    'Teknologi Informasi', 
    'Hubungan Masyarakat', 
    'Sales', 
    'Guru']

IndoBERT = 'indolem/indobert-base-uncased'
MODEL_PRE_TRAINED = IndoBERT

# MODEL_URL = 'https://cloud.wafi.web.id/cv_indo_model_v3.pth'
# MODEL_FOLDER = 'trained_model_online'
# MODEL_FILE = 'cv_indo_model_v3.pth'

# For testing purpose
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

def indo_predict_category(resume):
    is_success = False
    try:
        return predict_category(resume, tokenizer, model, class_labels)

    except Exception as e:
        return is_success, f'Error in prediction:  {str(e)}', 0