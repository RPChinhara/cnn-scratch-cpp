#include <chrono>
#include <thread>

#include "rand.h"
#include "tensor.h"

tensor synapse1 = glorot_uniform({32, 32});
tensor synapse2 = glorot_uniform({32, 32});
tensor synapse3 = glorot_uniform({32, 32});
tensor synapse4 = glorot_uniform({32, 32});

int main() {
    while (true) {
        std::cout << "Processing inputs...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));

        std::cout << "Backpropagating...\n\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    return 0;
}