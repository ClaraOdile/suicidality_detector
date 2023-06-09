import pandas as pd
from tqdm import tqdm
import neattext.functions as nfx
import tensorflow as tf
from transformers import DistilBertTokenizer

def clean_data(X:list, stopword=False) -> pd.DataFrame:
    '''
    - Post cleaning process
    - Depend on stopword option, stopwords will be removed
    '''
    # text_length=[]
    cleaned_texts=[]
    for sent in tqdm(X):
        sent=sent.lower()
        sent=nfx.remove_special_characters(sent)
        if not stopword:
            sent=nfx.remove_stopwords(sent)
        #text_length.append(len(sent.split()))
        cleaned_texts.append(sent)

    # df['clean_post_length'] = text_length
    print("✅ cleaning data is done \n")
    return cleaned_texts


def preprocessing(X:list, model_name='roberta-base'):
    tokenizer =  DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded = tokenizer(X, padding=True, truncation=True, return_tensors='tf')

    print("✅ tokenizing done \n")

    return encoded
