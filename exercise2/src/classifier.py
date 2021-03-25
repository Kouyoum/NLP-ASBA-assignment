
# Our modules
from resources.mybert         import MyBert
from resources.mydataset      import MyDataset
from resources.preprocessing  import *
from resources.training       import *

# transformers
import transformers
from transformers import AutoTokenizer, AutoModel

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader


# other Python modules
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
import numpy as np 
import os


# # weights
# weights = torch.tensor(traindata['sentiment'].value_counts())
# weights = weights / weights.sum()
# print(weights)
# weights = 1.0 / weights
# weights = weights / weights.sum()
# print(weights)

# Class variables
PRETRAIN_MODEL = "activebus/BERT_Review"
BATCH_SIZE = 25
EPOCHS = 1
MAX_LEN = 164
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier:
    """The Classifier"""

    def __init__(self):
        self.model = MyBert(model_name=PRETRAIN_MODEL, n_classes=3)
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL)
        self.best_accuracy = 0

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        
        # loading the data and preprocessing on the dataframe
        df = read_data(trainfile)
        df = prepare_dataframe(df)

        # Split Dataset and Dataloader
        df_train, df_val = split_dataframe(df, test_size = 0.1)
        
        train_data_loader = create_data_loader(df_train, self.tokenizer, BATCH_SIZE, max_len= MAX_LEN)
        val_data_loader = create_data_loader(df_val, self.tokenizer, BATCH_SIZE, max_len = MAX_LEN)

        # hyperparameters
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in tqdm(range(EPOCHS)):

            # Training 
            train_acc, train_loss = train_epoch(
                self.model,
                train_data_loader,    
                loss_fn, 
                optimizer, 
                device, 
                scheduler, 
                len(df_train)
            )

            # Validation
            val_acc, val_loss = eval_epoch(
                self.model,
                val_data_loader,
                loss_fn, 
                device, 
                len(df_val)
            )
            print("End Epoch {} \n".format(epoch + 1))
            if val_acc > self.best_accuracy:
                torch.save(self.model.state_dict(), 'models/best_model_state_.bin')
                self.best_accuracy = val_acc


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
        # Loading the model, setting up device and in eval mode
        self.model.load_state_dict(torch.load('models/best_model_state_.bin'))
        self.model.to(device)
        self.model.eval()
    
        # loading the data and preprocessing on the dataframe
        df = read_data(datafile)
        df = prepare_dataframe(df)
        
        # transform df in the adequate Dataset, to be fed to Bert
        dataset = MyDataset(
                reviews=df.X.to_numpy(),
                targets=df.y_true.to_numpy(),
                tokenizer=self.tokenizer,
                max_len=MAX_LEN
                    )

        predictions = []       # List where we store Bert's prediction
        with torch.no_grad():
            for d in dataset:
                texts = d["review_text"]
                input_ids = d["input_ids"][None, :].to(device)

                attention_mask = d["attention_mask"][None, :].to(device)
                targets = d["targets"].to(device)
            
                outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                predictions.append(preds)
                #real_values.append(targets)

        predictions = torch.tensor(predictions).numpy()

        # converting the digits back into the sentiments categories
        sentiments_dict = {0:"positive", 1:"negative", 2:"neutral"}
        predictions_sentiments = [sentiments_dict[i] for i in predictions]

        return predictions_sentiments




