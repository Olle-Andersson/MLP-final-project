import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def ams(y_pred, y, weights) -> np.float64:
    '''
        Calculate AMS score
    '''
    s = np.float64(weights[(y == 1) & (y_pred == 1)].sum())
    b = np.float64(weights[(y == 0) & (y_pred == 1)].sum())
    return np.sqrt(2 * ((s + b + 10) * np.log(1 + s / (b + 10)) - s))

def train_and_val_model(model, param_grid, train_df, val_df):
    '''
        Trains and validates model using the ams score.

        Returns the best model.
    '''
    rand_seed = 42

    features = [f for f in val_df.columns if f not in ["Label", "Weight"]]
    if "Weight" in train_df.columns:
        print("Weight column found in train_df, ajja bajja!.")
        return -1
    X_train = train_df[features].to_numpy()
    X_val = val_df[features].to_numpy()
    y_train = train_df["Label"].to_numpy()
    y_val = val_df["Label"].to_numpy()
    weights_val = val_df["Weight"].to_numpy()

    cross_val = StratifiedKFold(n_splits = 5,
                        shuffle = True,
                            random_state = rand_seed
                           )

    grid_search = GridSearchCV(model,
                               param_grid,
                               cv = cross_val,
                               scoring = "precision",
                               verbose = True,
                               refit = True,
                               n_jobs = -1
                               )

    print("Training model with param_grid: ")
    print(param_grid, "\n")

    grid_search.fit(X_train,y_train)
    
    print("Selected params:")
    print(grid_search.best_params_)

    print("\nCross val score:")
    print(f"{grid_search.best_score_:.3f}")

    y_pred = grid_search.predict(X_val)

    ams_score = ams(y_pred, y_val, weights_val)
    print(f"AMS score: {ams_score:.4f}")

    return grid_search


if __name__ == "__main__":
    y = np.array([1,1,0,0,0,1,0])
    y_pred = np.array([1,0,0,1,0,1,0])
    weights = np.array([0.5,1,0.4,1.5,0.7,0.3,0.4])

    #breakpoint()
    score = ams(y_pred, y, weights)
    print(score)
