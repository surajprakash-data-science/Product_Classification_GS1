import pandas as pd
import fasttext


def train_fasttext_model(training_data):
    model =fasttext.train_supervised(
        input=training_data,
        epoch=25,
        lr=1e-2,
        wordNgrams=4,
        verbose=2,
        loss= 'softmax'
    )
    return model

# FastText Training Script
if __name__ == "__main__":
    from data_ingestion import load_data, preprocess_data, preprocessing_fasttext, fastext_formatting

    # Example usage
    file_path = "../data/goldstandard_eng.csv"
    df = load_data(file_path)
    df = preprocess_data(df)
    df = preprocessing_fasttext(df)
    lvl1_fasttext_data = fastext_formatting(df, 'GS1_Level1_Category', 'text', 'lvl1_fasttext')
    lvl2_fasttext_data = fastext_formatting(df, 'GS1_Level2_Category', 'lvl2_input', 'lvl2_fasttext')
    lvl3_fasttext_data = fastext_formatting(df, 'GS1_Level3_Category', 'lvl3_input', 'lvl3_fasttext')

    print("FastText formatting successfull.")

    print(lvl1_fasttext_data.head())
    print(lvl2_fasttext_data.head())
    print(lvl3_fasttext_data.head())

    # Save formatted data to temporary files for FastText training
    lvl1_fasttext_data.to_csv("lvl1_fasttext.txt", index=False, header=False)
    lvl2_fasttext_data.to_csv("lvl2_fasttext.txt", index=False, header=False)
    lvl3_fasttext_data.to_csv("lvl3_fasttext.txt", index=False, header=False)

    print("FastText training data saved.")

    # Train FastText models
    train_fasttext_model("lvl1_fasttext.txt")   
    train_fasttext_model("lvl2_fasttext.txt")
    train_fasttext_model("lvl3_fasttext.txt")
    
    print("FastText models trained successfully.")

