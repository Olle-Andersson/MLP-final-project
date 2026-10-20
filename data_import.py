import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATADIR = "datasets/"

def load_and_split_data(FILENAME: str, SEP: str = ",") -> list[pd.DataFrame]:
    '''
    Loads, splits, and saves data to training and test .csv files.
    
    '''
    print(f"importing: {DATADIR+FILENAME}")
    df = pd.read_csv(filepath_or_buffer=DATADIR+FILENAME,sep=SEP)
    breakpoint()
    all_columns = df.columns
    df = df[[c for c in all_columns if c not in ["EventId", "KaggleSet", "KaggleWeight"]]]
    
    X = df[[c for c in df.columns if c != "Label"]]  
    y = df["Label"].map({'b': 0, 's': 1})

    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2, 
                                                        random_state=42,
                                                        stratify=y)

    df_train = X_train
    df_train['Label'] = y_train
    df_test = X_test
    df_test['Label'] = y_test
    
    train_df_path = DATADIR+"training.csv"
    test_df_path = DATADIR+"test.csv"
    print(f"Saving training data to: {train_df_path}")
    df_train.to_csv(train_df_path,index=False)
    print(f"Saving test data to: {test_df_path}")
    df_test.to_csv(test_df_path,index=False)

    return [df_train, df_test]

def load_data() -> list[pd.DataFrame]:
    '''
    Loads training and test DataFrames, or creates them if they dont exist.
    
    '''
    try:
        return [pd.read_csv(DATADIR+"training.csv"), pd.read_csv(DATADIR+"test.csv")]
    except Exception:
       return load_and_split_data("atlas-higgs-challenge-2014-v2.csv") 


def make_exploration_subset(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[c for c in df.columns if c != "Label"]]  
    y = df["Label"]

    tmp_X, df_X, tmp_y, df_y = train_test_split(X,y,
                                        test_size=0.2, 
                                        random_state=42,
                                        stratify=y)
    breakpoint()
    new_df = df_X
    new_df["Label"] = df_y

    return new_df

if __name__ == "__main__":
    df_train, df_test = load_data()

    exp_df = make_exploration_subset(df_train)
