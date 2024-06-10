1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - load_imdb() has to return a vectorized tensor like tf.keras.datasets.imdb.load_data().
    - Start working on text_vectorization() on imdb_test.csv.
      - text_vectorization() should not pad 0 as it is now. I haven't done truncation so that is fine?
    - In tf.keras.datasets.imdb.load_data() document it says "Words are ranked by how often they occur (in the training set)" so maybe create vocab only using training set?
    - Maybe add num_words like arg to this function as well like tf.keras.datasets.imdb.load_data() does?
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