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
constexpr size_t num_heads = 4;

tensor w_1 = glorot_uniform({d_model, 32}); // TODO: intermidiate size should be bigger than d_model?
tensor w_2 = glorot_uniform({32, d_model});

tensor b_1 = glorot_uniform({seq_len, 1});
tensor b_2 = glorot_uniform({seq_len, 1});

tensor multihead_attention(const tensor& x) {
    size_t batch_size = x.shape.front();
    size_t head_dim;

    if (num_heads == 1)
        head_dim = d_model;
    else
        head_dim = d_model / num_heads;

    tensor outputs = zeros({batch_size, seq_len, d_model});

    // TODO: These should be daclared at the top of translation unit as always, but I want this function to be impelmented in lyrs files. I don't know what to do at the moment. I think this function can use these params even if I move it to lyrs file.

    // TODO: I may need each weights and biases for number of heads
    tensor w_q = glorot_uniform({d_model, d_model});
    tensor w_k = glorot_uniform({d_model, d_model});
    tensor w_v = glorot_uniform({d_model, d_model});

    tensor b_q = glorot_uniform({d_model, 1}); // NOTE: For perf, daclare it with shape of (d_model, head_dim)?
    tensor b_k = glorot_uniform({d_model, 1});
    tensor b_v = glorot_uniform({d_model, 1});

    tensor w_o = glorot_uniform({d_model, d_model});

    // TODO: I want to make a operator extract a matrix from 3D or 4D tensor -> this is fundamentally same as slicing 3D/4D tensor to extract matrices so...
    // TODO: Should I modify matmul() to support 3D or even 4D tensors like NumPy does? There's no concept of 3D matrix multiplication in traditional math, so it would essentially be the same whether the 3D handling is done in matmul() or at this level. However, for now, handle it as I always do when dealing with 3D/4D tensors.

    // x: (10, 25, 128) or (8, 25, 128)
    // x_mat: (25, 128)

    for (size_t i = 0; i < batch_size; ++i) {
        tensor x_mat = slice(x, i * seq_len, seq_len);
        std::vector<tensor> attention_heads;

        tensor q_mat = matmul(x_mat, w_q);
        tensor k_mat = matmul(x_mat, w_k);
        tensor v_mat = matmul(x_mat, w_v);

        std::vector<std::vector<tensor>> heads(num_heads);

        // TODO: I think I need Multithreading for this?
        for (size_t j = 0; j < num_heads; ++j) {
            heads[j].push_back(q_mat.slice_cols(j * head_dim, (j + 1) * head_dim));
            heads[j].push_back(k_mat.slice_cols(j * head_dim, (j + 1) * head_dim));
            heads[j].push_back(v_mat.slice_cols(j * head_dim, (j + 1) * head_dim));

            tensor attention_scores = matmul(heads[j][0], transpose(heads[j][1]));
            tensor scaled_scores = attention_scores / sqrt(head_dim);
            tensor attention_weights = softmax(scaled_scores);

            // TODO: Add mask here to attention_weights?

            tensor weighted_sum = matmul(attention_weights, heads[j][2]);
            attention_heads.push_back(weighted_sum);
        }

        tensor concatenated_heads = concat(attention_heads, 1);
        tensor output = matmul(concatenated_heads, w_o);

        for (size_t j = 0; j < output.size; ++j)
            outputs[i * output.size + j] = output[j];
    }

    return outputs;
}

tensor encoder(const tensor& x) {
    tensor output = multihead_attention(x);
    tensor attention_output = layer_normalization(output + x);

    // attention_output: (10, 25, 128) or (8, 25, 128)

    size_t batch_size = x.shape.front();

    tensor outputs = zeros({batch_size, seq_len, d_model});

    // TODO: How to deal with matmul with shape 3D/4D? Do it as always or make new system?
    for (size_t i = 0; i < batch_size; ++i) {
        tensor attention_output_mat = slice(attention_output, i * seq_len, seq_len);
        tensor ffn = matmul(relu(matmul(attention_output_mat, w_1) + b_1), w_2) + b_2; // (25, 128) x (128, 32) x (32, 128) = (25, 128)
        tensor y = (ffn + attention_output_mat); // TODO: Add biases // (25, 128) x (25, 128)

        for (size_t j = 0; j < y.size; ++j)
            outputs[i * y.size + j] = y[j];
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