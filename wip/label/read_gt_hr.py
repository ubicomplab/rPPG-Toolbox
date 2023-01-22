import pandas as pd


def read_label(dataset):
    df = pd.read_csv("label/comparision_{0}.csv".format(dataset))
    out_dict = df.to_dict(orient='orient')
    return out_dict


