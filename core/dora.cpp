#include <iostream>
#include <thread>
#include <chrono>

#include "rand.h"
#include "tensor.h"

tensor w1 = glorot_uniform({32, 32});
tensor w2 = glorot_uniform({32, 32});
tensor w3 = glorot_uniform({32, 32});
tensor w4 = glorot_uniform({32, 32});

int main() {
    while (true) {
        std::cout << "Learning...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));  // Sleep for 10 seconds
    }
    return 0;
}