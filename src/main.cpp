#include "acts.h"
#include "arrs.h"
#include "datasets.h"
#include "linalg.h"
#include "lyrs.h"
#include "rand.h"

#include <chrono>

constexpr size_t vocab_size = 5000;
constexpr size_t seq_len = 25;
constexpr size_t d_model = 128; // NOTE: must be divisible by num_heads
constexpr size_t d_ff = 512; // NOTE: often 4x larger than d_model
constexpr size_t num_heads = 4;
size_t head_dim = (num_heads == 1) ? d_model : d_model / num_heads;

std::vector<std::vector<tensor>> w = {
    {
        glorot_uniform({d_model, head_dim}), // w_q
        glorot_uniform({d_model, head_dim}), // w_k
        glorot_uniform({d_model, head_dim})  // w_v
    },
    {
        glorot_uniform({d_model, head_dim}),
        glorot_uniform({d_model, head_dim}),
        glorot_uniform({d_model, head_dim})
    },
    {
        glorot_uniform({d_model, head_dim}),
        glorot_uniform({d_model, head_dim}),
        glorot_uniform({d_model, head_dim})
    },
    {
        glorot_uniform({d_model, head_dim}),
        glorot_uniform({d_model, head_dim}),
        glorot_uniform({d_model, head_dim})
    },
    {
        glorot_uniform({d_model, d_model}) // w_o
    }
};

tensor w_1 = glorot_uniform({d_model, d_ff});
tensor w_2 = glorot_uniform({d_ff, d_model});

tensor b_1 = glorot_uniform({1, d_ff}); // NOTE: Could be (seq_len, d_ff), but it'd be inefficient for memory specially when the seq_len, d_model, and d_ff get much bigger.
tensor b_2 = glorot_uniform({1, d_model});

tensor encoder(const tensor& x) {
    tensor output = multihead_attention(x, w, seq_len, d_model, num_heads);
    tensor attention_output = layer_normalization(output + x);

    size_t batch_size = x.shape.front();

    tensor outputs = zeros({batch_size, seq_len, d_model});

    for (size_t i = 0; i < batch_size; ++i) {
        tensor attention_output_mat = slice(attention_output, i * seq_len, seq_len);
        tensor ffn = matmul(relu(matmul(attention_output_mat, w_1) + b_1), w_2) + b_2;
        tensor y = layer_normalization(ffn + attention_output_mat);

        std::copy(y.elems, y.elems + y.size, outputs.elems + i * y.size);
    }

    return outputs;
}

tensor decoder(const tensor& x) {
    return tensor();
}

tensor train(const tensor& x_train, const tensor& y_train) {
    constexpr size_t epochs = 5;
    constexpr float lr = 0.01f;
    float batch_size = 10.0f;

    float num_samples = x_train.shape.front();
    const size_t num_batches = static_cast<size_t>(ceil(num_samples / batch_size));

    // NOTE: Embedding matrix is updated during backpropagation, similar to other model weights.
    auto embedding_lyr = embedding(vocab_size, d_model);
    auto positional_encoding_lyr = positional_encoding(seq_len, d_model);

    for (size_t i = 1; i <= epochs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Epoch " << i << "/" << epochs << "\n";

        float loss = 0.0f;

        // TODO: I have to process multiple batches simultaneously in order to speed up training lol That is why batach training is faster right?
        for (size_t j = 0; j < num_batches; ++j) {
            size_t start_idx = j * batch_size;
            size_t end_idx = std::min(start_idx + batch_size, num_samples);

            tensor x_batch = slice(x_train, start_idx, end_idx - start_idx);
            tensor y_batch = slice(y_train, start_idx, end_idx - start_idx);

            tensor embedded_tokens = embedding_lyr.adapt(x_batch);
            tensor input_embeddings = positional_encoding_lyr.adapt(embedded_tokens);

            // TODO: I run these functions simultaneously?
            tensor outputs = encoder(input_embeddings);
            tensor y = decoder(outputs);

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