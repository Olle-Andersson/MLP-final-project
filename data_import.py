import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATADIR = "datasets/"

def load_and_split_data(FILENAME: str, SEP: str = ",") -> list[pd.DataFrame]:
    '''
    Loads, splits, and saves data to training and test .csv files.
    
    RETURNS
    training dataFrame, test dataFrame
    '''

    print(f"importing: {DATADIR+FILENAME}")
    df = pd.read_csv(filepath_or_buffer=DATADIR+FILENAME,sep=SEP)
    all_columns = df.columns
    df = df[[c for c in all_columns if c not in ["KaggleSet", "KaggleWeight"]]]
    
    X = df[[c for c in df.columns if c != "Label"]]  
    y = df["Label"].map({'b': 0, 's': 1})

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2, 
                                                        random_state=42,
                                                        stratify=y)

    train_df = X_train
    train_df['Label'] = y_train
    test_df = X_test
    test_df['Label'] = y_test
    
    train_df_path = DATADIR+"training.csv"
    test_df_path = DATADIR+"test.csv"
    print(f"Saving training data to: {train_df_path}")
    train_df.to_csv(train_df_path)
    print(f"Saving test data to: {test_df_path}")
    test_df.to_csv(test_df_path)

    return [train_df, test_df]

def load_data():
    try:
        train_df = pd.read_csv(DATADIR+"training.csv")
        test_df = pd.read_csv(DATADIR+"test.csv")
    except Exception:
       train_df, test_df = load_and_split_data("atlas-higgs-challenge-2014-v2.csv") 
    
    X_train = train_df[[c for c in train_df.columns if c not in ["Label", "Weight"]]]  
    y_train = train_df["Label"].map({'b': 0, 's': 1})
    weights_train = train_df["Weight"]

    X_test = test_df[[c for c in test_df.columns if c not in ["Label", "Weight"]]]  
    y_test = test_df["Label"].map({'b': 0, 's': 1})
    weights_test = test_df["Weight"]

    return [
            X_train, 
            y_train.to_numpy().reshape(-1,1),
            weights_train.to_numpy().reshape(-1,1), 
            X_test, 
            y_test.to_numpy().reshape(-1,1), 
            weights_test.to_numpy().reshape(-1,1)
            ]
        

if __name__ == "__main__":
    X_train, y_train, weights_train, X_test, y_test, weights_test = load_data()
