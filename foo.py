# import pandas as pd
# import re

# import nltk
# nltk.download('punkt')

# from nltk.tokenize import word_tokenize

# data = pd.read_csv('datasets\\IMDB Dataset.csv')

# # text cleaning

# def rm_link(text):
#     return re.sub(r'https?://\S+|www\.\S+', '', text)

# # handle case like "shut up okay?Im only 10 years old"
# # become "shut up okay Im only 10 years old"
# def rm_punct2(text):
#     # return re.sub(r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)
#     return re.sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)

# def rm_html(text):
#     return re.sub(r'<[^>]+>', '', text)

# def space_bt_punct(text):
#     pattern = r'([.,!?-])'
#     s = re.sub(pattern, r' \1 ', text)     # add whitespaces between punctuation
#     s = re.sub(r'\s{2,}', ' ', s)        # remove double whitespaces    
#     return s

# def rm_number(text):
#     return re.sub(r'\d+', '', text)

# def rm_whitespaces(text):
#     return re.sub(r' +', ' ', text)

# def rm_nonascii(text):
#     return re.sub(r'[^\x00-\x7f]', r'', text)

# def rm_emoji(text):
#     emojis = re.compile(
#         '['
#         u'\U0001F600-\U0001F64F'  # emoticons
#         u'\U0001F300-\U0001F5FF'  # symbols & pictographs
#         u'\U0001F680-\U0001F6FF'  # transport & map symbols
#         u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
#         u'\U00002702-\U000027B0'
#         u'\U000024C2-\U0001F251'
#         ']+',
#         flags=re.UNICODE
#     )
#     return emojis.sub(r'', text)

# def spell_correction(text):
#     return re.sub(r'(.)\1+', r'\1\1', text)

# def clean_pipeline(text):    
#     no_link = rm_link(text)
#     no_html = rm_html(no_link)
#     space_punct = space_bt_punct(no_html)
#     no_punct = rm_punct2(space_punct)
#     no_number = rm_number(no_punct)
#     no_whitespaces = rm_whitespaces(no_number)
#     no_nonasci = rm_nonascii(no_whitespaces)
#     no_emoji = rm_emoji(no_nonasci)
#     spell_corrected = spell_correction(no_emoji)
#     return spell_corrected

# # clean_pipeline()
# print(data['review'][0])
# print()
# print(clean_pipeline(data['review'][0]))
# print(word_tokenize(clean_pipeline(data['review'][0])))

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the IMDb dataset
max_features = 10000  # Number of words to consider as features
maxlen = 200  # Cut texts after this number of words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

print(train_data)

# Pad sequences to make them uniform length
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Display dataset shapes
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print(train_data)