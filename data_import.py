import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def splitData(FILEPATH: str, SEP: str = ",") -> list[pd.DataFrame]:
    '''
    Loads, splits, and saves data to training and test .csv files.
    
    RETURNS
    training dataFrame, test dataFrame
    '''
    df = pd.read_csv(filepath_or_buffer=FILEPATH,sep=SEP)
    all_columns = df.columns
    df = df[[c for c in all_columns if c not in ["KaggleSet", "KaggleWeight"]]]
    
    X = df[[c for c in df.columns if c != "Label"]]  
    y = df["Label"].map({'b': 0, 's': 1})

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    train_df = X_train
    train_df['Label'] = y_train
    test_df = X_test
    test_df['Label'] = y_test
    
    train_df.to_csv("training.csv")
    test_df.to_csv("test.csv")

    return [train_df, test_df]

if __name__ == "__main__":
    splitData("./atlas-higgs-challenge-2014-v2.csv",)
