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

def preprocessing_fasttext(df):
    df["lvl2_input"] = df["GS1_Level1_Category"] + " " + df["text"]
    df["lvl3_input"] = df["GS1_Level1_Category"] + " " + df["GS1_Level2_Category"] + " " + df["text"]

    return df

def fastext_formatting(df, label_col, text_col, column_name):
    df_formatted = df[[label_col, text_col]].copy()
    df_formatted[label_col] = "__label__" + df_formatted[label_col].astype(str)
    df_formatted[column_name] = df_formatted[label_col] + " " + df_formatted[text_col]

    return df_formatted[column_name]