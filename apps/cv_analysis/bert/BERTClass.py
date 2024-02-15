import torch
import transformers
import torch
from transformers import BertTokenizer
import numpy as np
import requests
import os

class BERTClass(torch.nn.Module):
    def __init__(self, model_pre_trained):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_pre_trained)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 24)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output