#include "arrs.h"
#include "logger.h"
#include "renderer.h"
#include "window.h"
#include "tensor.h"

/*
TODOS:
    > Standard Transformer (Original)
    > Draw a rectangle
    - Variants of Transformers:
        - Stacked Transformer
        - RL model (DQN?)
        - Encoder-Only (BERT, RoBERTa, ALBERT)
        - Decoder-Only (GPT-2, GPT-3, GPT-4, ChatGPT, LLaMA)
        - Encoder-Decoder (T5, BART, mT5)
        - LLMs (Focus: English, science standards)
        - Vision-Language Models (DeepSeek, MoE)
        - Hybrid (Vision + NLP: CLIP, DINO, Flamingo)
    - Story Visualization
    - Autoencoder, GAN, YOLO, Mamba
    - Diffusion Transformers (DiT, UViT, DALL·E)
    - Vision Transformer (ViT, Swin, DeiT) [Dataset: MNIST?]
    - AlphaFold
    - AGI (Physics: Quantum gravity, time, entropy)

    - Brain-Inspired AI:
        - Neurogenesis: Add new synapses dynamically
        - Dynamic Synapses: Grow/shrink based on activity (neuroplasticity)
        - Stochastic Firing: Introduce noise for creativity & flexibility
        - Localized Updates: Each neuron adjusts its own weights
        - Graph-based AI: More biologically accurate than dense layers
        - Hebbian Learning: Alternative to backpropagation
        - Adaptive Networks: Add/remove connections dynamically
        - Neuro-Symbolic AI: Combine logic with deep learning
        - Spiking Neural Networks (SNNs): Biologically inspired learning

    - Memory & Forgetting:
        - Adaptive Forgetting: Remove irrelevant data over time
        - Memory Consolidation: Store long-term knowledge gradually
        - Attention-Based Recall: Retrieve only relevant memories
        - Emulate human memory (Working, Long-term, Episodic)
*/

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    logger::init();
    logger::log(fill({2, 3}, 1.0f));

    window window(hInstance);

    renderer renderer(window.get_hwnd());
    if (!renderer.init()) return -1;

    while (window.process_messages()) {
        renderer.render();
    }

    return 0;
}