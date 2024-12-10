#include "lyrs.h"

int main() {
    std::vector<std::string> vocab = {"foo bar", "bar baz", "baz bada boom"};
    std::vector<std::string> in = {"foo qux bar", "qux baz"};

    std::cout << text_vectorization(vocab, in, 5000, 4) << std::endl;

    return 0;
}