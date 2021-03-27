import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len=164):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
        review, 
        add_special_tokens=True, 
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
        padding='max_length',
        max_length=self.max_len
        )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }
