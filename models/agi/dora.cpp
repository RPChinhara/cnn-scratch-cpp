#include <chrono>
#include <thread>

#include "rand.h"
#include "tensor.h"

// TODO: Dynamic Synapses: Instead of fixed 32x32 matrices, let synapses grow or shrink based on activity (like neuroplasticity).
tensor synapse1 = glorot_uniform({32, 32});
tensor synapse2 = glorot_uniform({32, 32});
tensor synapse3 = glorot_uniform({32, 32});
tensor synapse4 = glorot_uniform({32, 32});

// NOTE: I don't need a memory since it's not stored in a single neuron or location—it is distributed across neurons and synapses in a network-like structure
// Right now, AI only has short-term recall (via attention mechanisms). Humans, however, have:

// Working memory (holds temporary thoughts).
// Long-term memory (consolidates knowledge).
// Episodic memory (stores past experiences).
// ✅ What to do?

// Implement memory consolidation: Store long-term knowledge gradually.
// Add attention-based recall: Retrieve only relevant memories.
// Use adaptive forgetting: Remove unimportant details over time.
// 🔹 Why? This allows AI to store knowledge persistently, like humans do.

// NOTE: Rougly, each neuron has 1162.79069767 synapses because human has ~86 billion neurons and ~100 trillion synapses so 100 trillion / 86 billion = 1162.79069767. These synapses are strengthen or weaken when new knowledge is stored similar to how weights and biases are updated during the backprop.

int main() {
    while (true) {
        std::cout << "Processing inputs...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));

        // TODO: Add new synapses (neurogenesis)
        // TODO: Rewires neurons (synapses) dynamically (neuroplasticity)
        // TODO: Stochastic firing: neurons fire randomly, even without direct input.
        //  - This adds "noise" to brain activity, helping with creativity, problem-solving, and flexibility.
        //  - It prevents the brain from getting stuck in fixed patterns (unlike AI, which follows strict mathematical rules).
        //  - AI models only activate when given input (no random self-firing).
        //  - AI doesn’t have spontaneous thought generation—it only predicts based on patterns.
        // TODO: Instead of full network updates using gradient-based weight updates, each neuron should updates its own weights.
        // TODO: Try dynamic neural architectures (where connections form and break like neuroplasticity).
        // TODO: Consider graph-based AI models instead of just dense layers as it more similar to how neurons in brain are connected?
        // TODO: Experiment with Hebbian learning rules instead of backpropagation.
        // TODO: Build a dynamic neural network that adds/removes connections like the brain.
        // TODO: Study Neuro-Symbolic AI to combine logic + deep learning.
        // TODO: Explore spiking neural networks (SNNs) for biologically accurate learning.
        // TODO: Modular Networks → Different specialized subnetworks for vision, language, reasoning, etc.
        // TODO: Use pc as enviroment for the model.
        // TODO: Implement adaptive forgetting (removing useless data to free memory).
        std::cout << "Backpropagating...\n\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    return 0;
}