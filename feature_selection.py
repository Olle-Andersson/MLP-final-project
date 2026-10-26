import pandas as pd
import numpy as np

def p_x(p: np.ndarray, phi: np.ndarray) -> np.ndarray:
    '''
        Calculate x-comp of momentum.
        Undefined if p < 0.
    '''
    comp = p * np.cos(phi)
    comp[(p < 0)] = -999

    return comp

def p_y(p: np.ndarray, phi: np.ndarray) -> np.ndarray:
    '''
        Calculate y-comp of momentum.
        Undefined if p < 0.
    '''
    comp = p * np.sin(phi)
    comp[(p < 0)] = -999

    return comp

def p_z(p: np.ndarray, eta: np.ndarray) -> np.ndarray:
    '''
        Calculate z-comp of momentum.
        Undefined if p < 0.
    '''
    comp = p * np.sinh(eta)
    comp[(p < 0)] = -999

    return comp

def p_t(px, py, pz):
    '''
        Calculate total momentum.
        Undefined if componenets are undefined.
    '''
    tot = np.sqrt( px ** 2 + py ** 2 + pz ** 2)
    tot[(px == -999)] = -999
    return tot

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Adds momentum components, total momentum, and removes eta, phi and 
        transverse momentum columns.
    '''
    prefixes = ["PRI_tau", 
                "PRI_lep", 
                "PRI_jet_leading", 
                "PRI_jet_subleading"]
    for prefix in prefixes:
        pt = df[prefix+"_pt"].to_numpy()
        phi = df[prefix+"_phi"].to_numpy()
        eta = df[prefix+"_eta"].to_numpy()

        df = df.drop(columns = [prefix+"_pt"])
        df = df.drop(columns = [prefix+"_phi"])
        df = df.drop(columns = [prefix+"_eta"])

        px = p_x(pt, phi)
        py = p_y(pt, phi)
        pz = p_z(pt, eta)
        p = p_t(px, py, pz)

        df[prefix+"_px"] = px
        df[prefix+"_py"] = py
        df[prefix+"_pz"] = pz
        df[prefix+"_p"] = p

    return df

if __name__ == "__main__":
    from data_import import load_data
    df_train, df_test = load_data()
    df_new = add_features(df_train)

    df_new.info()


