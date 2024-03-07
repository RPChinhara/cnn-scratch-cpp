import pandas as pd

# Assuming the dataset is in CSV format
imdb_data = pd.read_csv('dataset/imdb dataset.csv')

print(imdb_data)
print(imdb_data['review'][0])
print(imdb_data['sentiment'][0])

# Example: Convert text to lowercase
imdb_data['review'] = imdb_data['review'].apply(lambda x: x.lower())

print(imdb_data['review'][0])
print(imdb_data['sentiment'][0])