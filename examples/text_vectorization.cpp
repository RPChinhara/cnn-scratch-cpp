#include "lyrs.h"

#include <iostream>

int main() {
    const size_t max_tokens = 5000;
    const size_t max_len = 4;

    std::vector<std::string> vocab = {"foo bar", "bar baz", "baz bada boom"};
    std::vector<std::string> in = {"foo qux bar", "qux baz"};

    std::cout << text_vectorization(vocab, in, max_tokens, max_len) << std::endl;

    return 0;
}