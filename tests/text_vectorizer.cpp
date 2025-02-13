#include "lyrs.h"

int main() {
    std::vector<std::string> vocab = {"foo bar", "bar baz", "baz bada boom"};
    std::vector<std::string> in = {"foo qux bar", "qux baz"};

    size_t vocab_size = 5000;
    size_t seq_len = 4;

    text_vectorizer vectorizer(vocab_size, seq_len);
    vectorizer.build_vocab(vocab);

    std::cout << vectorizer.adapt(in) << "\n";

    return 0;
}