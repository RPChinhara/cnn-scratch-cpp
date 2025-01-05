- LeNet
    > Shape of kernel2 should be (16, 6, 5, 5) instead of (16, 5, 5)? The 6 channels in each filter contain different 5x5 kernels — each kernel corresponds to a different channel in the input.
         - I will fix convolution() so that it will support 4d kernel shape like follow (16, 6, 5, 5), but if I do this, how to get a shape (16, 6, 5, 5) for dl_dkernel2 in lenet_dl_dkernel2.cpp?
    - When I compute dc3_z/ds2, transpose the kernel2 into (6, 16, 5, 5) so that when convoluted with dl/dc3_z, it'd be (6, 14, 14) which is the shape of s2. Transpose looks like below. I don't know if it's true though.
                                  [1, 2  [13, 14
                                   3, 4], 15, 16]

    [1, 2  [5, 6  [9,  10         [5, 6  [17, 18
     3, 4], 7, 8], 11, 12]         7, 8], 19, 20]

    [13, 14  [17, 18  [21, 22     [9, 10  [21, 22
     15, 16], 19, 20], 23, 24] -> 11, 12], 23, 24]

- AlexNet (use ImageNet as the model was made for the dataset? It seems this is the way. -> I guess I can just use MNIST or other dataset that is less computation heavy so that I could proceed forward quickly)
- VGG
- ResNet
- Seq2seq
- Transformer
- Autoencoder
- BERT
- GAN
- BART
- Mamba
- Diffusion models
- YOLO
- Vision transformer
- Vision-Language Models
- AlphaFold
- DALL·E
- Quantum neural network
- Story Visualization
- Multimodal learning
- AGI (use a multimodal model to predict physical phenomena and solve unsolved problems, e.g., predict quantum gravity)