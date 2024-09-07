- No serious refactorizations untill I get to 5.

1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Implement SimpleRNN
    > Implement BPTT.
      - Update w_xh
      - Check if how I calculating BPTT is correct like if transpose is used correctly at right place...
      - Split into batch? Or make it adaptable to any batch size like I did for nn?
      - add validation dataset?
  - Implement LSTM.
  - Implement GRU.
  - Implement Bidirectional RNNs.
  - Implement Deep RNNs, which have multiple layers of RNNs stacked on top of each other and can be built with any of the basic RNN units (vanilla, LSTM, GRU).
  - Fix nn to use wx + b equation instead of current xw + b
  - Do I want to things like this by taking arguments in main()? ./ml_program --learning_rate 0.001 --epochs 100
  - Implement CNN.
  - Implement ConvLSTM.
2. Work on other tutorials on TensorFlow sites, e.g., Neural machine translation with a Transformer and Keras.
3. Implement other famous modles like BERT.
4. Implement interesting algorithms from latest papers.
5. Develop my own architectures, algorithms, and models.
6. Use 5 to solve existing problems and discover new theories in mathematics and physics.
7. Create new things, possibly scientific devices, from 6.