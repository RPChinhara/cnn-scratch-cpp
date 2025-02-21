#include <iostream>
#include <thread>
#include <chrono>

#include "rand.h"
#include "tensor.h"

tensor w1 = glorot_uniform({32, 32});
tensor w2 = glorot_uniform({32, 32});
tensor w3 = glorot_uniform({32, 32});
tensor w4 = glorot_uniform({32, 32});

// NOTE: Rougly, each neuron has 1162.79069767 synapses because human has ~86 billion neurons and ~100 trillion synapses so 100 trillion / 86 billion = 1162.79069767. These synapses are strengthen or weaken when new knowledge is stored similar to how weights and biases are updated during the backprop.

int main() {
    while (true) {
        std::cout << "Learning...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));  // Sleep for 10 seconds
    }
    return 0;
}