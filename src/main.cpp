#include "arrs.h"
#include "datasets.h"
#include "lyrs.h"

#include <chrono>

constexpr size_t vocab_size = 5000;
constexpr size_t max_len = 25;

constexpr size_t embedding_dim = 50;

tensor forward(const tensor& x, float batch_size) {
    return tensor();
}

tensor train(const tensor& x_train, const tensor& y_train) {
    constexpr size_t epochs = 10;
    constexpr float lr = 0.01f;
    float batch_size = 64.0f;

    const size_t num_batches = static_cast<size_t>(ceil(60000.0f / batch_size));

    for (size_t i = 1; i <= epochs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Epoch " << i << "/" << epochs << "\n";

        float loss = 0.0f;

        batch_size = 64.0f;

        // TODO: I have to process multiple batches simultaneously in order to speed up training lol That is why batach training is faster right?
        for (size_t j = 0; j < num_batches; ++j) {
            size_t start_idx = j * batch_size; // 937 x 64 = 59968
            size_t end_idx = std::min(start_idx + batch_size, 60000.0f);

            tensor x_batch = slice_4d(x_train, start_idx, end_idx - start_idx);
            tensor y_batch = slice(y_train, start_idx, end_idx - start_idx);

            if (j == num_batches - 1)
                batch_size = static_cast<float>(end_idx - start_idx);

            tensor y = forward(x_batch, batch_size);

            // loss = categorical_cross_entropy(y_batch, y);

            // Backpropagation

            if (j == num_batches - 1)
                std::cout << "\r\033[K";
            else
                std::cout << "\r\033[K" << j + 1 << "/" << num_batches << " - loss: " << loss << std::flush;
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << num_batches << "/" << num_batches << " - " << std::chrono::duration<double>(end - start).count() << "s/step - loss: " << 0.0f << "\n";
    }

    return tensor();
}

float evaluate(const tensor& x_test, const tensor& y_test) {
    return 0.0f;
}

tensor predict(const tensor& x_test, const tensor& y_test) {
    return tensor();
}

int main() {

    auto a = variable({2, 3}, {1, 2, 3, 1, 2, 3});
    auto v = embedding(10, 3, a);

    std::cout << v.mat << "\n";
    std::cout << v.dense_vecs << "\n";

    // auto input_target = load_daily_dialog("datasets/daily_dialog/daily_dialog.csv");
    // auto input = load_daily_dialog("datasets/daily_dialog/daily_dialog_input.csv");
    // auto target = load_daily_dialog("datasets/daily_dialog/daily_dialog_target.csv");

    // // TODO: If I make text_vectorization() a class, runtime will be 1/2 of now as I only need to create the vocabulary once for "input_target". I don't need to do it twice.
    // // TODO: I may need to use subword tokenizers for better results. I'm using a simple tokenizer.
    // tensor input_token = text_vectorization(input_target, input, vocab_size, max_len);
    // tensor target_token = text_vectorization(input_target, target, vocab_size, max_len);

    // auto input_token_train_test = split(input_token, 0.2f);
    // auto target_token_train_test = split(target_token, 0.2f);

    // train(input_token_train_test.first, target_token_train_test.first);

    // std::cout << "Test loss: " << evaluate(input_token_train_test.second, target_token_train_test.second) << "\n\n";

    // tensor test_predictions = predict(input_token_train_test.second, target_token_train_test.second);

    return 0;
}