#include "arrs.h"
#include "datasets.h"
#include "lyrs.h"

#include <chrono>

constexpr size_t vocab_size = 5000;
constexpr size_t seq_len = 25;
constexpr size_t model_dim = 5;

tensor encoder(const tensor& x, float batch_size) {
    tensor x_norm = layer_normalization(x);

    // Multiheaded attention
    return tensor();
}

tensor decoder(const tensor& x, float batch_size) {
    return tensor();
}

// TODO: Move to lyrs.h since it's one of the layer
tensor positional_encoding(const size_t seq_len, const size_t dim) {
    tensor output = zeros({seq_len, dim});

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < dim / 2; ++j) {
            float denominator = pow(10000, 2.0f * j / dim);
            output(i, 2 * j) = sin(i / denominator);
            output(i, 2 * j + 1) = cos(i / denominator);
        }
    }

    return output;
}

tensor train(const tensor& x_train, const tensor& y_train) {
    constexpr size_t epochs = 5;
    constexpr float lr = 0.01f;
    float batch_size = 10.0f;

    float num_samples = x_train.shape.front();
    const size_t num_batches = static_cast<size_t>(ceil(num_samples / batch_size));

    for (size_t i = 1; i <= epochs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Epoch " << i << "/" << epochs << "\n";

        float loss = 0.0f;

        batch_size = 10.0f;

        // TODO: I have to process multiple batches simultaneously in order to speed up training lol That is why batach training is faster right?
        for (size_t j = 0; j < num_batches; ++j) {
            size_t start_idx = j * batch_size;
            size_t end_idx = std::min(start_idx + batch_size, num_samples);

            tensor x_batch = slice(x_train, start_idx, end_idx - start_idx);
            tensor y_batch = slice(y_train, start_idx, end_idx - start_idx);

            embedding embedding_lyr = embedding(5000, model_dim, x_batch); // TODO: I think embedding() and positional_encoding() should be called before epoch for loop?
            tensor position_encoded_tesnor = positional_encoding(seq_len, model_dim);

            // Adding embeddings and po position_encoded_tesnor
            size_t idx = 0;
            for (size_t k = 0; k < embedding_lyr.dense_vecs.size; ++k) {
                if (k == embedding_lyr.dense_vecs.shape[1] * embedding_lyr.dense_vecs.shape[2])
                    idx = 0;
                embedding_lyr.dense_vecs[k] = embedding_lyr.dense_vecs[k] + position_encoded_tesnor[idx];
                ++idx;
            }

            if (j == num_batches - 1)
                batch_size = static_cast<float>(end_idx - start_idx);

            tensor outputs = encoder(x_batch, batch_size); // TODO: I may not need to change batch size as this was only required in the CNN
            tensor y = decoder(outputs, batch_size); // TODO: I may not need to change batch size as this was only required in the CNN

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
    // OPTIMIZE: If I make load_daily_dialog() return input and target by utlizing " in the datset, I only need to call this function once which improve performance a lot.
    // auto input_target = load_daily_dialog("datasets/daily_dialog/daily_dialog.csv");
    // auto input = load_daily_dialog("datasets/daily_dialog/daily_dialog_input.csv");
    // auto target = load_daily_dialog("datasets/daily_dialog/daily_dialog_target.csv");

    // OPTIMIZE: If I make text_vectorization() a class, runtime will be 1/2 of now as I only need to create the vocabulary once for "input_target". I don't need to do it twice.
    // TODO: I may need to use subword tokenizers for better results. I'm using a simple tokenizer.
    // tensor input_token = text_vectorization(input_target, input, vocab_size, seq_len);
    // tensor target_token = text_vectorization(input_target, target, vocab_size, seq_len);

    tensor dammy_input_token = zeros({60, seq_len});
    tensor dammy_target_token = fill({60, seq_len}, 2.0f);

    auto input_token_train_test = split(dammy_input_token, 0.2f);
    auto target_token_train_test = split(dammy_target_token, 0.2f);

    train(input_token_train_test.first, target_token_train_test.first);

    std::cout << "Test loss: " << evaluate(input_token_train_test.second, target_token_train_test.second) << "\n\n";

    tensor test_predictions = predict(input_token_train_test.second, target_token_train_test.second);

    return 0;
}