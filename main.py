# pip install "tensorflow-text>=2.11"
# pip install einops

# import numpy as np

# import typing
# from typing import Any, Tuple

# import einops
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import tensorflow as tf
# import tensorflow_text as tf_text

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# class ShapeChecker():
#   def __init__(self):
#     # Keep a cache of every axis-name seen
#     self.shapes = {}

#   def __call__(self, tensor, names, broadcast=False):
#     if not tf.executing_eagerly():
#       return

#     parsed = einops.parse_shape(tensor, names)

#     for name, new_dim in parsed.items():
#       old_dim = self.shapes.get(name, None)

#       if (broadcast and new_dim == 1):
#         continue

#       if old_dim is None:
#         # If the axis name is new, add its length to the cache.
#         self.shapes[name] = new_dim
#         continue

#       if new_dim != old_dim:
#         raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
#                          f"    found: {new_dim}\n"
#                          f"    expected: {old_dim}\n")

# # Download the file
# import pathlib

# path_to_zip = tf.keras.utils.get_file(
#     'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
#     extract=True)

# path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng/spa.txt'

# def load_data(path):
#   text = path.read_text(encoding='utf-8')

#   lines = text.splitlines()
#   pairs = [line.split('\t') for line in lines]

#   context = np.array([context for target, context in pairs])
#   target = np.array([target for target, context in pairs])

#   return target, context

# target_raw, context_raw = load_data(path_to_file)

# for x in range(2):
#   print(target_raw[x])
#   print(context_raw[x], '\n')

# print(len(context_raw[True]))
# print(context_raw[False])

# print(target_raw)
# print(len(target_raw))
# print(len(context_raw))

# BUFFER_SIZE = len(context_raw)
# BATCH_SIZE = 64

# is_train = np.random.uniform(size=(len(target_raw),)) < 1.0

# train_raw = (
#     tf.data.Dataset
#     .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
#     .shuffle(BUFFER_SIZE)
#     .batch(BATCH_SIZE))
# val_raw = (
#     tf.data.Dataset
#     .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
#     .shuffle(BUFFER_SIZE)
#     .batch(BATCH_SIZE))

# # for element in train_raw:
# #   print(element)

# print(is_train)
# train_size = sum(is_train)
# val_size = sum(~is_train)
# print("Size of train_raw:", train_size)
# print("Size of train_raw:", val_size)
# print(95192 + 23772)

# for example_context_strings, example_target_strings in train_raw.take(4):
#   print(example_context_strings[:66])
#   print()
#   print(example_target_strings[:5])
#   break

# example_text = tf.constant('Ve.')

# print(example_text.numpy())
# print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())

# def tf_lower_and_split_punct(text):
#   # Split accented characters.
#   text = tf_text.normalize_utf8(text, 'NFKD')
#   text = tf.strings.lower(text)
#   # Keep space, a to z, and select punctuation.
#   text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
#   # Add spaces around punctuation.
#   text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
#   # Strip whitespace.
#   text = tf.strings.strip(text)

#   text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
#   return text

# print(example_text.numpy().decode())
# print(tf_lower_and_split_punct(example_text).numpy().decode())

# max_vocab_size = 5000

# context_text_processor = tf.keras.layers.TextVectorization(
#     standardize=tf_lower_and_split_punct,
#     max_tokens=max_vocab_size,
#     ragged=True)

# context_text_processor.adapt(train_raw.map(lambda context, target: context))

# # Here are the first 10 words from the vocabulary:
# print(context_text_processor.get_vocabulary()[:])
# print(context_text_processor.vocabulary_size())

# input_data = ["foo qux bar qux baz dog sex sex sex sex"]
# input_data2 = [["Ve."], ["Vete."], ["Vaya."], ["Tomátelo con soda."]]
# print(context_text_processor(input_data))
# print(context_text_processor(input_data2).to_tensor())
# print(context_text_processor(input_data2))

# target_text_processor = tf.keras.layers.TextVectorization(
#     standardize=tf_lower_and_split_punct,
#     max_tokens=max_vocab_size,
#     ragged=True)

# target_text_processor.adapt(train_raw.map(lambda context, target: target))
# print(target_text_processor.get_vocabulary()[:])
# print(target_text_processor.vocabulary_size())

# input_data = ["foo qux bar qux baz dog sex"]
# print(target_text_processor(input_data)[:,:-1])
# print(target_text_processor(input_data)[:,1:])
# print(target_text_processor(input_data)[:,:])
# print(target_text_processor(input_data))

# example_tokens = context_text_processor(example_context_strings)
# print(example_tokens[:30, :])

# example_tokens2 = context_text_processor(example_text)
# print(example_tokens2)

# context_vocab = np.array(context_text_processor.get_vocabulary())
# tokens = context_vocab[example_tokens[0].numpy()]
# ' '.join(tokens)

# plt.subplot(1, 2, 1)
# plt.pcolormesh(example_tokens.to_tensor())
# plt.title('Token IDs')

# plt.subplot(1, 2, 2)
# plt.pcolormesh(example_tokens.to_tensor() != 0)
# plt.title('Mask')

# def process_text(context, target):
#   context = context_text_processor(context).to_tensor()
#   target = target_text_processor(target)
#   targ_in = target[:,:-1].to_tensor()
#   targ_out = target[:,1:].to_tensor()
#   return (context, targ_in), targ_out


# train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
# val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

# for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
#   print(ex_context_tok[0, :].numpy())
#   print()
#   print(ex_tar_in[0, :].numpy())
#   print(ex_tar_out[0, :].numpy())

# UNITS = 256

# class Encoder(tf.keras.layers.Layer):
#   def __init__(self, text_processor, units):
#     super(Encoder, self).__init__()
#     self.text_processor = text_processor
#     self.vocab_size = text_processor.vocabulary_size()
#     self.units = units

#     # The embedding layer converts tokens to vectors
#     self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
#                                                mask_zero=True)

#     # The RNN layer processes those vectors sequentially.
#     self.rnn = tf.keras.layers.Bidirectional(
#         merge_mode='sum',
#         layer=tf.keras.layers.GRU(units,
#                             # Return the sequence and state
#                             return_sequences=True,
#                             recurrent_initializer='glorot_uniform'))

#   def call(self, x):
#     shape_checker = ShapeChecker()
#     shape_checker(x, 'batch s')

#     # 2. The embedding layer looks up the embedding vector for each token.
#     x = self.embedding(x)
#     shape_checker(x, 'batch s units')

#     # 3. The GRU processes the sequence of embeddings.
#     x = self.rnn(x)
#     shape_checker(x, 'batch s units')

#     # 4. Returns the new sequence of embeddings.
#     return x

#   def convert_input(self, texts):
#     texts = tf.convert_to_tensor(texts)
#     if len(texts.shape) == 0:
#       texts = tf.convert_to_tensor(texts)[tf.newaxis]
#     context = self.text_processor(texts).to_tensor()
#     context = self(context)
#     return context

# # Encode the input sequence.
# encoder = Encoder(context_text_processor, UNITS)
# ex_context = encoder(ex_context_tok)

# print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
# print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')

# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# # Parameters
# vocab_size = 10000  # Only consider the top 10,000 words in the dataset
# max_length = 200    # Pad sequences to a maximum length of 200
# embedding_dim = 50  # Dimension of word embeddings
# hidden_units = 256  # Number of units in the RNN layer
# output_size = 1     # Binary classification (positive or negative)

# # Load and preprocess the IMDb dataset
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# # Pad sequences to ensure equal length
# X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
# X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

# # Build the model
# model = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
#     SimpleRNN(hidden_units),
#     Dense(output_size, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print the model summary
# model.summary()

# # Train the model
# model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test Accuracy: {accuracy:.4f}')

# # Example prediction
# sample_review = X_test[0]  # Using the first test sample
# prediction = model.predict(tf.expand_dims(sample_review, 0))
# print(f'Predicted sentiment (0 = negative, 1 = positive): {prediction[0][0]:.4f}')

# import tensorflow as tf

# # (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)
# (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(seed=113)

# # Inspect the first encoded review
# print("Encoded review:", train_data[0])
# print("Encoded review:", train_data[1])
# print("Encoded review:", train_data[2])

# # Decode the first review back to words for verification
# word_index = tf.keras.datasets.imdb.get_word_index()
# reverse_word_index = {value: key for (key, value) in word_index.items()}

# for i in range(30):
#   decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[i]])
#   print("Decoded review:", decoded_review)

# import tensorflow as tf

# # Load the IMDB dataset
# (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data()

# # Get the word index dictionary
# word_index = tf.keras.datasets.imdb.get_word_index()

# # Reverse the word index dictionary to get a mapping from indices to words
# reverse_word_index = {value: key for (key, value) in word_index.items()}

# # Print out the vocabulary
# print("Vocabulary size:", len(word_index))
# print("Some sample words from the vocabulary:")
# for i, (word, index) in enumerate(word_index.items()):
#     if i >= 20:  # Only print the first 20 words for demonstration
#         break
#     print(f"{word}: {index}")

# # If you want to include padding (index 0), start (index 1), and unknown (index 2) tokens:
# reverse_word_index_with_reserved = {value + 3: key for (key, value) in word_index.items()}
# reverse_word_index_with_reserved[0] = "<PAD>"
# reverse_word_index_with_reserved[1] = "<START>"
# reverse_word_index_with_reserved[2] = "<UNKNOWN>"
# reverse_word_index_with_reserved[3] = "<UNUSED>"

# print("\nVocabulary with reserved tokens:")
# for i in range(20):  # Only print the first 20 indices for demonstration
#     print(f"{i}: {reverse_word_index_with_reserved.get(i, '?')}")

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# # Parameters
# vocab_size = 10000  # Only consider the top 10,000 words in the dataset
# max_length = 200    # Pad sequences to a maximum length of 200
# embedding_dim = 50  # Dimension of word embeddings
# hidden_units = 256  # Number of units in the RNN layer
# output_size = 1     # Binary classification (positive or negative)

# # Load and preprocess the IMDb dataset
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# print(X_train[0])
# print(X_train[1])
# print(X_train[2])

# for i in range (3):
#   my_array = np.array(X_train[i])
#   print(my_array.shape)  # This will work

# # Pad sequences to ensure equal length
# X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
# X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

# print(X_train[0].shape)
# print(X_train[1].shape)
# print(X_train[2].shape)
# print(X_train[0])
# print(X_train[1])
# print(X_train[2])

max_tokens = 5000  # Maximum vocab size.
max_len = 7  # Sequence length to pad the outputs to.
# Create the layer.
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    # output_sequence_length=max_le,
    ragged=False)

# Now that the vocab layer has been created, call `adapt` on the
# list of strings to create the vocabulary.
vectorize_layer.adapt(["foo bar", "bar baz", "baz bada boom"])

# Now, the layer can map strings to integers -- you can use an
# embedding layer to map these integers to learned embeddings.
input_data = [["foo qux bar"], ["qux baz"]]
print(vectorize_layer(input_data))