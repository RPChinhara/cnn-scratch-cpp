#include "arrs.h"
#include "datasets.h"
#include "lyrs.h"

constexpr size_t vocab_size = 5000;
constexpr size_t max_len = 25;

constexpr size_t epochs = 250;
constexpr float lr = 0.01f;
size_t batch_size = 0;

constexpr size_t embedding_dim = 50;

int main() {
    auto input_target = load_daily_dialog("datasets/daily_dialog/daily_dialog.csv");
    auto input = load_daily_dialog("datasets/daily_dialog/daily_dialog_input.csv");
    auto target = load_daily_dialog("datasets/daily_dialog/daily_dialog_target.csv");

    tensor input_token = text_vectorization(input_target, input, vocab_size, max_len);
    tensor target_token = text_vectorization(input_target, target, vocab_size, max_len);

    auto input_token_train_test = split(input_token, 0.2f);
    auto target_token_train_test = split(target_token, 0.2f);

    return 0;
}