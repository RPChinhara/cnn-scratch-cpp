- No serious refactorizations untill I get to 5.

1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Implement SimpleRNN
    - Vanishing gradients might be occuring -> I'm sure it's happening as when I logged all the gradients, it was all close to zeros or zeros
      - Solutions
        - Better Activation Functions: Using ReLU or its variants can help mitigate this issue.
        - Batch Normalization: Normalizing the input to each layer can help maintain the scale of gradients.
        - Residual Connections: Architectures like ResNets allow gradients to flow through skip connections, helping to maintain stronger gradient signals.
        - Gradient Clipping: Preventing gradients from becoming too small or too large can also help.
        - Use different optimizers like Adam, RMSprop
        - Simplify the model
    - Now that I have new min_max_scaler class just use it
    > Is calculating and logging the losses comes before the BPTT?
      I've searched and the sequnece is:
      Forward pass -> Loss Calculation -> Logging the loss -> BPTT
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
3. Implement other famous modles like BERT.
4. Implement interesting algorithms from latest papers.
5. Research about history, and implement old networks like Hopfield network?
6. Develop my own architectures, algorithms, and models. Maybe tip is like Hopfield did use tools from physics, biology, chemistry, and so on as these represents/explains the nature. For instance, Newton's laws of motion?
7. Use 5 to solve existing problems and discover new theories in mathematics and physics.
8. Create new things, possibly scientific devices, from 6.