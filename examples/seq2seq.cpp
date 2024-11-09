#include "lyrs.h"

#include <fstream>
#include <sstream>

std::vector<std::string> daily_dialog(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file.is_open())
        std::cerr << "Failed to open the file: " << file_path << std::endl;

    std::vector<std::string> data;

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value);

        value = lower(value);
        value = regex_replace(value, "[.,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", " ");
        value = regex_replace(value, "\"", "");
        value = regex_replace(value, "\\s*[^\\x00-\\x7f]\\s*", "");
        value = regex_replace(value, "[^\\x00-\\x7f]", "");
        // value = regex_replace(value, "'", "");
        value = regex_replace(value, "\\s+", " ");
        value = regex_replace(value, "\\s+$", "");
        value = value.insert(0, "[START] ");

        data.push_back(value);
    }

    file.close();

    return data;
}

int main() {
    auto input_target = daily_dialog("datas/daily_dialog/daily_dialog.csv");
    auto input = daily_dialog("datas/daily_dialog/daily_dialog_input.csv");
    auto target = daily_dialog("datas/daily_dialog/daily_dialog_target.csv");

    size_t vocab_size = 5000;
    size_t max_len = 25;

    auto input_token = text_vectorization(input_target, input, vocab_size, max_len);
    auto target_token = text_vectorization(input_target, target, vocab_size, max_len);

    auto input_token_train_test = split(input_token, 0.2f);
    auto target_token_train_test = split(target_token, 0.2f);

    // TODO: Check vocab.size() as it may not be 5000
    gru model = gru(0.01f, vocab_size);
    model.train(input_token_train_test.first, target_token_train_test.first);

    return 0;
}