from tqdm import tqdm
import neattext.functions as nfx

'''
preprocess functions will take 'Post' column of Dataset and return cleaned_text and the length of text.

how to use :
df['cleaned_post'], df['cleaned_post_length'] = preprocess(df.Post)
'''

def preprocess(post):
    text_length = []
    cleaned_text = []
    for sent in tqdm(post):
        sent = sent.lower()
        sent = nfx.remove_special_characters(sent)
        sent = nfx.remove_stopwords(sent)
        text_length.append(len(sent.split()))
        cleaned_text.append(sent)
    return cleaned_text, text_length
