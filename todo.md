- No serious refactorizations untill I get to 5.

1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Implement SimpleRNN
    - Implement BPTT.
      - Recalculate w_hh
        > Calculate dL2/dh1
      - Recalculate w_xh
      - Evaluate which calculation for "d_loss_d_w_hh" reduces the loss more. Which is correct both of them?
        -
          8317 50 -> d_loss_d_h_t
          50 8317 -> 1.0f - sqrt(h_y.first.back())
          50 8317 -> h_y.first[h_y.first.size() - 1]

          Currently, the calculation is matmul((transpose(d_loss_d_h_t) * (1.0f - sqrt(h_y.first.back()))), transpose(h_y.first[h_y.first.size() - 1])).
          However, it could also be matmul(1.0f - sqrt(h_y.first.back()), transpose(h_y.first[h_y.first.size() - 1])) which will be shape of (50, 50). sum(d_loss_d_h_t, 0) which turns into shape of (1, 50).
          Finally, you add this (1, 50) to first result.
        -
      - Split into batch? Or make it adaptable to any batch size like I did for nn?
      - add validation dataset?
  - Implement either one-to-many or many-to-many
    - Implement whichever is more famous so that I could learn how loss would work and its derivatives
  - Implement LSTM.
  - Implement GRU.
  - Implement Bidirectional RNNs.
  - Implement Deep RNNs, which have multiple layers of RNNs stacked on top of each other and can be built with any of the basic RNN units (vanilla, LSTM, GRU).
  - Fix nn to use wx + b equation instead of current xw + b
    (64, 10) -> (64, 1) or (64, 10) I think latter is clearer, but
    former is more performant. (10, 64) -> (1, 64) x.T = (4, 10), w1 = (64, 4), w2 = (10(must), 64), w3 = (64, 3),
    output = (64, 3) x = (10, 4), w1 = (4, 64), w2 = (64, 64), w3 = (64, 3), ouput = (10, 3)
  - Do I want to things like this by taking arguments in main()? ./ml_program --learning_rate 0.001 --epochs 100
  - Implement CNN.
  - Implement ConvLSTM.
2. Work on other tutorials on TensorFlow sites, e.g., Neural machine translation with a Transformer and Keras.
3. Implement other famous modles like BERT.
4. Implement interesting algorithms from latest papers.
5. Develop my own architectures, algorithms, and models.
6. Use 5 to solve existing problems and discover new theories in mathematics and physics.
7. Create new things, possibly scientific devices, from 6.