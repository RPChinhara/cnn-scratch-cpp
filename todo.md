- No serious refactorizations untill I get to 5.

1. Working on a tutorial https://www.tensorflow.org/text/tutorials/nmt_with_attention
  - Implement SimpleRNN
    - Implement BPTT.
      > 1. Try batch size of 1 as the loss was really good with it.
        - The way caluclate the loss is wrong if I'm using batch size of like 32.
          First, devide total train dataset size by number batch size e.g., 80 / 32 = [2.5] = 3 batches.
          Then, in each epoch calculate losses in this case 3 times, and then accumulate these numbers and devide by 3 to get an average.
          And this is the loss I log for each epochs.
          Example calculation:
          If you have 3 batches with losses of 0.5, 0.3, and 0.4:
          Total Loss: total_loss = 0.5 + 0.3 + 0.4 = 1.2
          Average Loss: average_loss = 1.2 / 3 = 0.4
      2. If loss seems good recheck if the whole BPTT make sense
        - Recheck if calculation for d_loss_d_b_h is correct.
      3. Make it adaptable so that I can use different batch sizes like 32?
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