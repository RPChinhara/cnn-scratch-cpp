#include "acts.h"
#include "arrs.h"
#include "datasets.h"
#include "linalg.h"
#include "lyrs.h"
#include "rand.h"

#include <chrono>

constexpr size_t vocab_size = 5000;
constexpr size_t seq_len = 25;
constexpr size_t model_dim = 5;
constexpr size_t num_heads = 4;

tensor multihead_attention(const tensor& x) {
    constexpr size_t head_dim = 3;
    size_t batch_size = x.shape.front();
    size_t idx = 0;

    // TODO: These should be daclared at the top of translation unit as always, but I want this function to be impelmented in lyrs files. I don't know what to do at the moment.
    tensor w_q = glorot_uniform({model_dim, head_dim});
    tensor w_k = glorot_uniform({model_dim, head_dim});
    tensor w_v = glorot_uniform({model_dim, head_dim});

    tensor b_q = glorot_uniform({model_dim, head_dim});
    tensor b_k = glorot_uniform({model_dim, head_dim});
    tensor b_v = glorot_uniform({model_dim, head_dim});

    tensor q = zeros({batch_size, seq_len, head_dim});
    tensor k = zeros({batch_size, seq_len, head_dim});
    tensor v = zeros({batch_size, seq_len, head_dim});

    // TODO: I want to make a operator extract a matrix from 3D or 4D tensor -> this is fundamentally same as slicing 3D/4D tensor to extract matrices so...
    // TODO: Should I modify matmul() to support 3D or even 4D tensors like NumPy does? There's no concept of 3D matrix multiplication in traditional math, so it would essentially be the same whether the 3D handling is done in matmul() or at this level. However, for now, handle it as I always do when dealing with 3D/4D tensors.

    for (size_t i = 0; i < batch_size; ++i) {
        tensor x_mat = slice(x, i * seq_len, seq_len);

        tensor q_mat = matmul(x_mat, w_q);
        tensor k_mat = matmul(x_mat, w_k);
        tensor v_mat = matmul(x_mat, w_v);

        // Compute Attention Scores (Scaled Dot-Product Attention)
        tensor attention_scores = matmul(q_mat, transpose(k_mat));
        tensor scaled_scores = attention_scores / sqrt(head_dim);
        tensor attention_weights = softmax(scaled_scores);

        // Compute the Weighted Sum (Apply Attention)
        tensor output = matmul(attention_weights, v_mat);

        std::cout << output.get_shape() << "\n";

        for (size_t j = 0; j < q_mat.size; ++j) {
            q[idx * q_mat.size + i] = q_mat[i];
            k[idx * q_mat.size + i] = k_mat[i];
            v[idx * q_mat.size + i] = v_mat[i];
        }
    }

    return tensor();
}

tensor encoder(const tensor& x) {
    tensor x_norm = layer_normalization(x);
    tensor output = multihead_attention(x_norm);

    return tensor();
}

tensor decoder(const tensor& x) {
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

    embedding embedding_lyr = embedding(vocab_size, model_dim);

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

            // TODO: I think positional_encoding() should be called before epoch for loop?
            tensor position_encoded_tensor = positional_encoding(seq_len, model_dim);

            // TODO: Should I do this inside the positional_encoding()?
            // Adding embeddings and po position_encoded_tesnor
            size_t idx = 0;
            const size_t block_size = embedded_tokens.shape[1] * embedded_tokens.shape[2];
            
            for (size_t k = 0; k < embedded_tokens.size; ++k) {
                embedded_tokens[k] += position_encoded_tensor[idx];
                idx = (k + 1) % block_size == 0 ? 0 : idx + 1;
            }

            // TODO: I run these functions simultaneously?
            tensor outputs = encoder(embedded_tokens);
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