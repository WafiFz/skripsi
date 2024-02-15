import torch
import requests
import os
import numpy as np
from transformers import BertTokenizer

def download_model(MODEL_FOLDER, MODEL_FILE, MODEL_URL):
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    # Mendapatkan nama file dari URL
    file_name = os.path.join(MODEL_FOLDER, MODEL_FILE)

    # Memeriksa apakah file sudah ada
    if not os.path.exists(file_name):
        # Mengunduh file dari URL
        response = requests.get(MODEL_URL)

        # Menyimpan file ke folder lokal
        with open(file_name, 'wb') as file:
            file.write(response.content)

        print(f"Model downloaded and saved successfully at: {file_name}")
    else:
        print(f"Model.pth file already exists at: {file_name}")

    return file_name

def define_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def define_tokenizer(MODEL_PRE_TRAINED):
    return BertTokenizer.from_pretrained(MODEL_PRE_TRAINED)

def predict_category(resume, tokenizer, model, class_labels):
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