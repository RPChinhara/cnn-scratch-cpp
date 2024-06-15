1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Load_imdb() has to return a vectorized tensor like tf.keras.datasets.imdb.load_data().
    - Start working on text_vectorization() on imdb_test.csv.
      - Instead of using pad_sequences() control padding and truncation with args in text_vectorization() I think this is the modern way
        - Creatre num_words as an arg for load_imdb() and pass that to text_vectorization as max_token?
    - In tf.keras.datasets.imdb.load_data() document it says "Words are ranked by how often they occur (in the training set)" so maybe create vocab only using training set?
  - I have to get good performance or prediction on imdb with a model made by tf first.
  - Implement Vanilla RNNs, which are the simplest form of RNNs. They have only a single hidden layer.
    - Implement forward propagation for many-to-one and many-to-many, as these are more common. One-to-one and one-to-many are less common. One-to-one might just be a regular neural network when you think about it...
    - Implement backpropagation for all the cases mentioned above.
  - Implement LSTM.
  - Implement GRU.
  - Implement Bidirectional RNNs.
  - Implement Deep RNNs, which have multiple layers of RNNs stacked on top of each other and can be built with any of the basic RNN units (vanilla, LSTM, GRU).
2. Implement CNN.
3. Work on other tutorials on TensorFlow sites, e.g., Neural machine translation with a Transformer and Keras.
4. Implement other famous modles like BERT, ConvLSTM.
5. Implement interesting algorithms from papers.
6. Come up with your own algorithms and models learned from papers.
7. Solve and create new Mathematics and Physics problem theories using autoregressive transformer auto-generative type models.
8. Create new things/devices from 6.