1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  > ten x = zeros({150, 4}); should be data.x = zeros({150, 4}); in load_iris() I guess.
  - Add [START] and [END] words to load_imdb()?
  - Try the preprocess inside while loop when reading the imdb file to improve perf.
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
7. Use the best models, possibly autoregressive transformer models, to solve existing problems and discover new theories in mathematics and physics.
8. Create new things, possibly scientific devices, from 7.