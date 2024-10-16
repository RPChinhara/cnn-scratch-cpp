1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Implement SimpleRNN
    - Tackle the vanishing gradients
      - Implement Adam
    - Maybe use glorot_uniform in nn as well like in Tensorflow Dense?
    - Stop taking x_val and y_val? Try using validation datasets on google colab, and see if test loss is enough to decide it is not either over/under fitting. Essentially good loss.
    - Now shapes for b_h and b_y are (hidden_size, batch_size), and (output_size, batch_size) respectively, I have to change batch_size to 1 so that the shape would much when I run evaluate() which
      takes dataset like x_test which has different batch_size than training. This is what did exactly on nn.
    - Now that I have new min_max_scaler class just use it
      - In order for me to run evaluate() and predict() which mostly likely I'd pass test dataset which have different sizes than train dataset, I have to change "batch_size" in
        forward(). What'd work for temporaly is take enum Phase in forward() which contains TRAIN and TEST, and switch batch_size based on these enum. By the way, batch_size is = 8317 when the enum is
        TRAIN, and 2072 when it's TEST.
    - If the loss is as low as loss on google colab (successefully implemented SimpleRNN!) recheck if the whole BPTT make sense
      - Especially recheck if calculation for d_loss_d_b_h is correct as I've never searched and compare with result from chatGPT
      - is derivative of relu right?
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