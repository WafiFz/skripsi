import torch
import transformers
import torch
from transformers import BertTokenizer
import numpy as np
import requests
import os

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
PRE_TRAINED_MODEL = BERT

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(PRE_TRAINED_MODEL)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 24)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# Load the trained model
model = BERTClass()

# Path to file model

url_model = "https://wasteless.my.id/cv_model_v5.pth"
local_folder = "trained_model_online"

if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# Mendapatkan nama file dari URL
file_name = os.path.join(local_folder, "cv_model_v5.pth")

# Memeriksa apakah file sudah ada
if not os.path.exists(file_name):
    # Mengunduh file dari URL
    response = requests.get(url_model)

    # Menyimpan file ke folder lokal
    with open(file_name, 'wb') as file:
        file.write(response.content)

    print(f"Model berhasil diunduh dan disimpan di: {file_name}")
else:
    print(f"File model.pth sudah ada di: {file_name}")

# Set the model path
model_path = file_name

# Check is CUDA available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use torch.load woth map_location
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

def predict_category(resume):
    is_success = False
    try:
        # Tokenize the input resume text
        inputs = tokenizer.encode_plus(
            resume,
            None,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True
        )

        ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
        mask = torch.tensor(inputs['attention_mask']).unsqueeze(0)
        token_type_ids = torch.tensor(inputs['token_type_ids']).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            output = model(ids, mask, token_type_ids)

        # Process the output
        probabilities = torch.sigmoid(output)

        max_value_probabilities = torch.max(probabilities).item()

        # if max_value_probabilities < 0.5:
        #     raise ValueError("Probability less than 50%, the contents of the CV are not appropriate")

        # Get index class MAX probability
        predicted_index = np.argmax(probabilities, axis=1).item()

        # Get label class from index
        predicted_label = class_labels[predicted_index]

        is_success = True

        return is_success, predicted_label, max_value_probabilities

    except Exception as e:
        return is_success, f'Error in prediction:  {str(e)}', 0