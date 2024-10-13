1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Implement SimpleRNN
    - Taccle the vanishing gradients
      > Use relu, if possible leaky_relu
      - Implement Adam
    - Now that I have new min_max_scaler class just use it
    - If the loss is as low as loss on google colab (successefully implemented SimpleRNN!) recheck if the whole BPTT make sense
      - Especially recheck if calculation for d_loss_d_b_h is correct as I've never searched and compare with result from chatGPT
  - Implement LSTM.
    - Implement either one-to-many or many-to-many (Do whichever is more famous so that I could learn how loss would work and its derivatives)
  - Implement GRU.
  - Implement Bidirectional RNNs.
  - Implement Deep RNNs, which have multiple layers of RNNs stacked on top of each other and can be built with any of the basic RNN units (vanilla, LSTM, GRU).
  - Implement CNN.
  - Implement ConvLSTM.
2. Work on other tutorials on TensorFlow sites, e.g., Neural machine translation with a Transformer and Keras.
3. Implement algorithms from latest/famous/interesting papers.
4. Develop my own architectures, algorithms, and models. Maybe tip is like Hopfield did use tools from physics, biology, chemistry, and so on as these represents/explains the nature. For instance, Newton's laws of motion?
5. Use '4' to solve existing problems and discover new theories in mathematics and physics.
6. Create new things, possibly scientific devices, from '5'.