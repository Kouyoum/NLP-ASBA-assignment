import pandas as pd


def read_data(path):
  df = pd.read_csv(path, sep='\t', header=None, 
                    names=["sentiment", "aspect_category","target_term", "time", "review"])
  
  return df

def prepare_reviews(df):
  df["aspect_target"] = df.aspect_category + "#" + df.target_term
  df["y_true"] = df.sentiment.map({"positive": 0, "negative": 1, "neutral": 2})
  df["X"] = df.review + "[SEP]" + df.aspect_target + "[SEP]"
  return df[["X", "y_true"]]
