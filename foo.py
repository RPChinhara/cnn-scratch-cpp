import pandas as pd
import re

data = pd.read_csv('datasets\\IMDB Dataset.csv')
print(data.head())


def rm_nonascii(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)

def space_bt_punct(text):
    pattern = r'([.,!?-])'
    s = re.sub(pattern, r' \1 ', text)     # add whitespaces between punctuation
    # s = re.sub(r'\s{2,}', ' ', s)        # remove double whitespaces    
    return s

print(space_bt_punct("Hello,world!This is a-test...and some punctuation?"))