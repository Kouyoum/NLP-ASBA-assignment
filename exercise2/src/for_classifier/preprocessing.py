from for_classifier.mydataset      import MyDataset
from torch.utils.data         import DataLoader
from sklearn.model_selection  import train_test_split
import pandas as pd

def read_data(path):
    df = pd.read_csv(path, sep='\t', lineterminator='\n', header=None, 
                      names=["sentiment", "aspect_category","target_term", "time", "review"])
    
    #df.review = df.review.apply(lambda x: x.strip("\r"))
    return df

def prepare_dataframe(df):
    """
    1. Concatenates the aspect category and aspect target with the review.
    2. Changes the sentiments to integers.
    """
    df["aspect_target"] = df.aspect_category + "#" + df.target_term
    df["y_true"] = df.sentiment.map({"positive": 0, "negative": 1, "neutral": 2})
    df["X"] = df.review + "[SEP]" + df.aspect_target + "[SEP]"
    return df[["X", "y_true"]]

def split_dataframe(df, test_size=0.1):
    df_train, df_val = train_test_split(
    df,
    test_size=test_size,
    random_state=42)
    return df_train, df_val


def create_data_loader(df, tokenizer, batch_size, max_len=164, num_workers=0):
    ds = MyDataset(
      reviews=df.X.to_numpy(),
      targets=df.y_true.to_numpy(),
      tokenizer=tokenizer,
      max_len=max_len
    )
    return DataLoader(ds,batch_size=batch_size,num_workers=num_workers)