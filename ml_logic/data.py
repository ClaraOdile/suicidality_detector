import pandas as pd
from tqdm import tqdm
import neattext.functions as nfx
from transformers import DistilBertTokenizer

def preprocessing(X:list, stopword=False):
    cleaned_texts=[]
    for sent in X:
        sent=sent.lower()
        sent=nfx.remove_special_characters(sent)
        if stopword:
            sent=nfx.remove_stopwords(sent)
        cleaned_texts.append(sent)

    print("✅ cleaning data is done \n")
    tokenizer =  DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    print("✅ tokenizing started \n")
    encoded_texts = tokenizer(cleaned_texts, padding=True, truncation=True)

    print("✅ tokenizing done \n")

    return encoded_texts
