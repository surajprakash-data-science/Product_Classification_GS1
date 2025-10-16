import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):

    df.drop_duplicates(inplace=True)
    df.drop(columns=['GeneratedID', 'EntityNodeID', 'URL', 'HOST', 'Properties', 's:breadcrumb'], inplace=True)
    df.dropna(subset=['GS1_Level1_Category'], inplace=True)
    df.dropna(subset=['s:description'], inplace=True)
    df['s:brand'] = df['s:brand'].fillna('not available')
    df["text"] = (df["s:name"] + " " + df["s:description"] + " " + df["s:brand"]).str.strip()

    return df