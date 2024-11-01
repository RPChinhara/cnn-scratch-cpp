#include "lyrs.h"

// # Sample data
// questions = [
//     "Hello!",
//     "How are you?",
//     "What is your name?",
//     "Tell me a joke.",
//     "Goodbye!"
// ]

// answers = [
//     "Hi there!",
//     "I'm just a bunch of code, but I'm doing well!",
//     "I'm a chatbot created with TensorFlow!",
//     "Why don't scientists trust atoms? Because they make up everything!",
//     "See you later!"
// ]

// std::vector<std::string> vocab = {"foo bar", "bar baz", "baz bada boom"};
// std::vector<std::string> in = {"foo qux bar", "qux baz"};

// std::cout << text_vectorization(vocab, in, 5000, 4) << std::endl;

float mean_squared_error(const tensor &y_true, const tensor &y_pred) {
    float sum = 0.0f;
    float n = static_cast<float>(y_true.shape.back());

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / n;
}

int main() {


    gru model = gru(mean_squared_error, 0.01f);
    // model.train(x_y_train.first, x_y_train.second);

    std::cout << 1 << std::endl;

    return 0;
}