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

// NOTE: Rougly, each neuron has 1162.79069767 synapses because human has ~86 billion neurons and ~100 trillion synapses so 100 trillion / 86 billion = 1162.79069767. These synapses are strengthen or weaken when new knowledge is stored similar to how weights and biases are updated during the backprop.

int main() {
    while (true) {
        std::cout << "Processing inputs...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
        // TODO: Add new synapses (neurogenesis)
        // TODO: Rewires neurons (synapses) dynamically (neuroplasticity)
        std::cout << "Backpropagating...\n\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    return 0;
}