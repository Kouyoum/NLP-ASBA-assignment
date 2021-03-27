
from transformers import AutoTokenizer, AutoModel

import torch.nn as nn
#import torch.nn.functional as F


class MyBert(nn.Module):
    """Full model"""
    def __init__(self, model_name="activebus/BERT_Review", n_classes=3):
        super(MyBert, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    
    def forward(self, input_ids, attention_mask):
        _,output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
        #x = output["pooler_output"]
        x = self.dropout(output)
        x = self.linear(x)
  
        return nn.functional.softmax(x, dim=1)