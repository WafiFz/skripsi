import torch
import transformers
import torch
from transformers import BertTokenizer
import numpy as np

class_labels = ['accountant', 'advocate', 'agriculture', 'apparel', 'arts', 'automobile', 'aviation', 'banking', 'bpo', 'business-development', 'chef', 'construction', 'consultant', 'designer', 'digital-media', 'engineering', 'finance', 'fitness', 'healthcare', 'hr', 'information-technology', 'public-relations', 'sales', 'teacher']

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
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
model_path = "trained_model/cv_model_v4.pth"

# Check is CUDA available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use torch.load woth map_location
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

        if max_value_probabilities < 0.5:
            raise ValueError("Probability less than 50%, the contents of the CV are not appropriate")

        # Get index class MAX probability
        predicted_index = np.argmax(probabilities, axis=1).item()

        # Get label class from index
        predicted_label = class_labels[predicted_index]

        is_success = True

        return is_success, predicted_label, max_value_probabilities

    except Exception as e:
        return is_success, f'Error in prediction:  {str(e)}'