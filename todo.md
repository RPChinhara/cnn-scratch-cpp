1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Implement SimpleRNN
    > Implement BPTT.
      > I think h_t should be only initialized to zero at the very beginning, after that I should use last h_t generated in forward()
      - Check this for (auto j = 0; j < batch_size - seq_length; ++j) is right
      - why is it printing tensors only up to certain decimal places?
  - Implement LSTM.
  - Implement GRU.
  - Implement Bidirectional RNNs.
  - Implement Deep RNNs, which have multiple layers of RNNs stacked on top of each other and can be built with any of the basic RNN units (vanilla, LSTM, GRU).
  - Fix nn to use wx + b equation instead of current xw + b
2. Implement CNN.
3. Work on other tutorials on TensorFlow sites, e.g., Neural machine translation with a Transformer and Keras.
4. Implement other famous modles like BERT, ConvLSTM.
5. Implement interesting algorithms from latest papers.
6. Develop my own architectures, algorithms, and models.
7. Use 7 to solve existing problems and discover new theories in mathematics and physics.
8. Create new things, possibly scientific devices, from 7.